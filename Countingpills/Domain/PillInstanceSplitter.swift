import CoreGraphics
import CoreVideo
import Foundation

final class PillInstanceSplitter {
    private let modelSide: CGFloat
    private let maxRegionsToSplit = 3
    private let maxSplitPerRegion = 3
    private let maxGainPerRegion = 2

    init(modelSide: CGFloat) {
        self.modelSide = modelSide
    }

    func refine(detections: [PillDetection], in pixelBuffer: CVPixelBuffer) -> [PillDetection] {
        guard detections.count >= 2 else { return detections }
        guard let grayscale = GrayscaleImage(pixelBuffer: pixelBuffer) else { return detections }

        let sides = detections.map(\.meanSide).sorted()
        let medianSide = sides[sides.count / 2]
        let largeCutoff = max(22 as CGFloat, medianSide * 1.35)

        let suspects = detections
            .filter { $0.meanSide >= largeCutoff && $0.variantHits >= 2 && $0.avgScore >= 0.45 }
            .sorted { $0.meanSide > $1.meanSide }

        guard !suspects.isEmpty else { return detections }

        var refined = detections
        for suspect in suspects.prefix(maxRegionsToSplit) {
            guard let splits = splitMergedDetection(suspect, image: grayscale) else { continue }

            let capped = Array(splits.prefix(maxSplitPerRegion))
            let gain = capped.count - 1
            guard gain > 0, gain <= maxGainPerRegion else { continue }

            let removeRadius = max(0.01, min(0.05, (suspect.meanSide / modelSide) * 0.36))
            if let removeIndex = refined.firstIndex(where: { normalizedDistance($0.point, suspect.point) <= removeRadius }) {
                refined.remove(at: removeIndex)
            }
            refined.append(contentsOf: capped)
        }

        return deduplicate(refined)
    }

    private func splitMergedDetection(_ suspect: PillDetection, image: GrayscaleImage) -> [PillDetection]? {
        let centerX = Int((suspect.point.x * modelSide).rounded())
        let centerY = Int((suspect.point.y * modelSide).rounded())

        let side = Int(max(100 as CGFloat, min(280 as CGFloat, suspect.meanSide * 3.1)).rounded())
        let half = side / 2

        let roi = IntRect(
            x: max(0, centerX - half),
            y: max(0, centerY - half),
            width: min(side, image.width - max(0, centerX - half)),
            height: min(side, image.height - max(0, centerY - half))
        )

        guard roi.width >= 72, roi.height >= 72 else { return nil }

        let patch = image.crop(roi)
        let threshold = otsuThreshold(pixels: patch.pixels)

        let brightMask = patch.pixels.map { $0 >= threshold ? UInt8(1) : UInt8(0) }
        let darkMask = patch.pixels.map { $0 <= threshold ? UInt8(1) : UInt8(0) }

        guard let selected = selectForegroundMask(
            brightMask: brightMask,
            darkMask: darkMask,
            width: patch.width,
            height: patch.height,
            suspect: suspect
        ) else {
            return nil
        }

        let dt = distanceTransform(mask: selected.mask, width: patch.width, height: patch.height)
        let minPeakDistance = max(6, Int((suspect.meanSide * 0.22).rounded()))
        let peakFloor = max(10, Int((suspect.meanSide * 0.12).rounded()) * 3)

        let peaks = findPeaks(
            distance: dt,
            mask: selected.mask,
            width: patch.width,
            height: patch.height,
            minDistance: minPeakDistance,
            floor: peakFloor,
            maxCount: 4
        )

        guard peaks.count >= 2 else { return nil }

        let instances = assignPixelsToPeaks(
            peaks: peaks,
            mask: selected.mask,
            width: patch.width,
            height: patch.height
        )

        let minClusterArea = max(18, Int((suspect.meanSide * suspect.meanSide * 0.08).rounded()))
        let valid = instances.filter { $0.count >= minClusterArea }
        guard valid.count >= 2 else { return nil }

        let mapped: [PillDetection] = valid.prefix(maxSplitPerRegion).map { cluster in
            let gx = (CGFloat(roi.x) + cluster.centroid.x) / modelSide
            let gy = (CGFloat(roi.y) + cluster.centroid.y) / modelSide
            let meanSide = sqrt(CGFloat(cluster.count)) * 1.3
            return PillDetection(
                point: CGPoint(x: gx.clamped(to: 0...1), y: gy.clamped(to: 0...1)),
                meanSide: meanSide,
                avgScore: max(0.5, suspect.avgScore * 0.9),
                variantHits: max(2, suspect.variantHits)
            )
        }

        let separation = minimumSeparation(mapped.map(\.point))
        guard separation > 0.008 else { return nil }
        return mapped
    }

    private func selectForegroundMask(
        brightMask: [UInt8],
        darkMask: [UInt8],
        width: Int,
        height: Int,
        suspect: PillDetection
    ) -> SelectedComponent? {
        let bright = evaluateMask(
            brightMask,
            width: width,
            height: height,
            suspect: suspect,
            expectedMergedArea: suspect.meanSide * suspect.meanSide * 1.7
        )
        let dark = evaluateMask(
            darkMask,
            width: width,
            height: height,
            suspect: suspect,
            expectedMergedArea: suspect.meanSide * suspect.meanSide * 1.7
        )

        switch (bright, dark) {
        case let (b?, d?):
            return b.score >= d.score ? b : d
        case let (b?, nil):
            return b
        case let (nil, d?):
            return d
        default:
            return nil
        }
    }

    private func evaluateMask(
        _ mask: [UInt8],
        width: Int,
        height: Int,
        suspect: PillDetection,
        expectedMergedArea: CGFloat
    ) -> SelectedComponent? {
        let opened = dilate(mask: erode(mask: mask, width: width, height: height), width: width, height: height)
        let closed = erode(mask: dilate(mask: opened, width: width, height: height), width: width, height: height)

        let components = connectedComponents(mask: closed, width: width, height: height)
        guard !components.isEmpty else { return nil }

        let center = CGPoint(x: CGFloat(width) * 0.5, y: CGFloat(height) * 0.5)

        var best: SelectedComponent?
        for component in components {
            let area = CGFloat(component.count)
            if area < 20 || area > CGFloat(width * height) * 0.86 { continue }

            let dist = hypot(component.centroid.x - center.x, component.centroid.y - center.y)
            let normDist = dist / max(1, sqrt(CGFloat(width * width + height * height)))
            let areaScore = 1 - min(1, abs(log(max(0.001, area / max(1, expectedMergedArea)))))
            let score = areaScore * 1.4 - normDist * 1.2

            let selected = SelectedComponent(mask: componentMask(component: component, total: width * height), score: score)
            if best == nil || score > best!.score {
                best = selected
            }
        }

        return best
    }

    private func componentMask(component: Component, total: Int) -> [UInt8] {
        var mask = [UInt8](repeating: 0, count: total)
        for i in component.indices where i >= 0 && i < total {
            mask[i] = 1
        }
        return mask
    }

    private func connectedComponents(mask: [UInt8], width: Int, height: Int) -> [Component] {
        var labels = [Int](repeating: -1, count: mask.count)
        var components: [Component] = []
        components.reserveCapacity(8)

        var queue = [Int](repeating: 0, count: mask.count)

        for start in 0..<mask.count {
            if mask[start] == 0 || labels[start] != -1 { continue }

            var head = 0
            var tail = 0
            queue[tail] = start
            tail += 1
            labels[start] = components.count

            var indices: [Int] = []
            indices.reserveCapacity(256)

            var sumX: CGFloat = 0
            var sumY: CGFloat = 0

            while head < tail {
                let current = queue[head]
                head += 1

                indices.append(current)
                let x = current % width
                let y = current / width
                sumX += CGFloat(x)
                sumY += CGFloat(y)

                for ny in max(0, y - 1)...min(height - 1, y + 1) {
                    for nx in max(0, x - 1)...min(width - 1, x + 1) {
                        let ni = ny * width + nx
                        if mask[ni] == 0 || labels[ni] != -1 { continue }
                        labels[ni] = components.count
                        queue[tail] = ni
                        tail += 1
                    }
                }
            }

            let count = indices.count
            guard count > 0 else { continue }
            let centroid = CGPoint(x: sumX / CGFloat(count), y: sumY / CGFloat(count))
            components.append(Component(indices: indices, count: count, centroid: centroid))
        }

        return components
    }

    private func distanceTransform(mask: [UInt8], width: Int, height: Int) -> [Int] {
        let inf = 1_000_000
        var dist = [Int](repeating: 0, count: mask.count)

        for i in 0..<mask.count {
            dist[i] = mask[i] == 1 ? inf : 0
        }

        for y in 0..<height {
            for x in 0..<width {
                let i = y * width + x
                if mask[i] == 0 { continue }

                var best = dist[i]
                if x > 0 { best = min(best, dist[i - 1] + 3) }
                if y > 0 { best = min(best, dist[i - width] + 3) }
                if x > 0 && y > 0 { best = min(best, dist[i - width - 1] + 4) }
                if x + 1 < width && y > 0 { best = min(best, dist[i - width + 1] + 4) }
                dist[i] = best
            }
        }

        for y in stride(from: height - 1, through: 0, by: -1) {
            for x in stride(from: width - 1, through: 0, by: -1) {
                let i = y * width + x
                if mask[i] == 0 { continue }

                var best = dist[i]
                if x + 1 < width { best = min(best, dist[i + 1] + 3) }
                if y + 1 < height { best = min(best, dist[i + width] + 3) }
                if x + 1 < width && y + 1 < height { best = min(best, dist[i + width + 1] + 4) }
                if x > 0 && y + 1 < height { best = min(best, dist[i + width - 1] + 4) }
                dist[i] = best
            }
        }

        return dist
    }

    private func findPeaks(
        distance: [Int],
        mask: [UInt8],
        width: Int,
        height: Int,
        minDistance: Int,
        floor: Int,
        maxCount: Int
    ) -> [Peak] {
        var peaks: [Peak] = []
        peaks.reserveCapacity(16)

        for y in 1..<(height - 1) {
            for x in 1..<(width - 1) {
                let i = y * width + x
                if mask[i] == 0 { continue }
                let v = distance[i]
                if v < floor { continue }

                var isPeak = true
                for ny in (y - 1)...(y + 1) {
                    for nx in (x - 1)...(x + 1) {
                        if nx == x && ny == y { continue }
                        if distance[ny * width + nx] > v {
                            isPeak = false
                            break
                        }
                    }
                    if !isPeak { break }
                }

                if isPeak {
                    peaks.append(Peak(x: x, y: y, value: v))
                }
            }
        }

        peaks.sort { $0.value > $1.value }

        var selected: [Peak] = []
        selected.reserveCapacity(maxCount)
        let minD2 = minDistance * minDistance

        for peak in peaks {
            var tooClose = false
            for existing in selected {
                let dx = peak.x - existing.x
                let dy = peak.y - existing.y
                if dx * dx + dy * dy < minD2 {
                    tooClose = true
                    break
                }
            }
            if tooClose { continue }

            selected.append(peak)
            if selected.count >= maxCount {
                break
            }
        }

        return selected
    }

    private func assignPixelsToPeaks(
        peaks: [Peak],
        mask: [UInt8],
        width: Int,
        height: Int
    ) -> [InstanceCluster] {
        guard !peaks.isEmpty else { return [] }

        var count = [Int](repeating: 0, count: peaks.count)
        var sumX = [CGFloat](repeating: 0, count: peaks.count)
        var sumY = [CGFloat](repeating: 0, count: peaks.count)

        for y in 0..<height {
            for x in 0..<width {
                let i = y * width + x
                if mask[i] == 0 { continue }

                var bestIndex = 0
                var bestDistance = Int.max

                for (idx, peak) in peaks.enumerated() {
                    let dx = x - peak.x
                    let dy = y - peak.y
                    let d2 = dx * dx + dy * dy
                    if d2 < bestDistance {
                        bestDistance = d2
                        bestIndex = idx
                    }
                }

                count[bestIndex] += 1
                sumX[bestIndex] += CGFloat(x)
                sumY[bestIndex] += CGFloat(y)
            }
        }

        var clusters: [InstanceCluster] = []
        clusters.reserveCapacity(peaks.count)

        for idx in 0..<peaks.count where count[idx] > 0 {
            clusters.append(
                InstanceCluster(
                    count: count[idx],
                    centroid: CGPoint(
                        x: sumX[idx] / CGFloat(count[idx]),
                        y: sumY[idx] / CGFloat(count[idx])
                    )
                )
            )
        }

        return clusters
    }

    private func deduplicate(_ detections: [PillDetection]) -> [PillDetection] {
        guard detections.count > 1 else { return detections }

        let sorted = detections.sorted {
            if $0.variantHits != $1.variantHits { return $0.variantHits > $1.variantHits }
            if $0.avgScore != $1.avgScore { return $0.avgScore > $1.avgScore }
            return $0.meanSide > $1.meanSide
        }

        let sides = sorted.map(\.meanSide).sorted()
        let medianSide = sides[sides.count / 2]
        let radius = max(0.007, min(0.028, (medianSide / modelSide) * 0.20))

        var kept: [PillDetection] = []
        kept.reserveCapacity(sorted.count)

        for det in sorted {
            if kept.contains(where: { normalizedDistance($0.point, det.point) <= radius }) {
                continue
            }
            kept.append(det)
        }

        return kept
    }

    private func minimumSeparation(_ points: [CGPoint]) -> CGFloat {
        guard points.count >= 2 else { return 0 }

        var minDistance = CGFloat.greatestFiniteMagnitude
        for i in 0..<(points.count - 1) {
            for j in (i + 1)..<points.count {
                minDistance = min(minDistance, normalizedDistance(points[i], points[j]))
            }
        }
        return minDistance == .greatestFiniteMagnitude ? 0 : minDistance
    }

    private func normalizedDistance(_ a: CGPoint, _ b: CGPoint) -> CGFloat {
        let dx = a.x - b.x
        let dy = a.y - b.y
        return sqrt(dx * dx + dy * dy)
    }

    private func otsuThreshold(pixels: [UInt8]) -> UInt8 {
        guard !pixels.isEmpty else { return 128 }

        var histogram = [Int](repeating: 0, count: 256)
        for value in pixels {
            histogram[Int(value)] += 1
        }

        let total = pixels.count
        var sum: Double = 0
        for i in 0..<256 {
            sum += Double(i * histogram[i])
        }

        var sumB: Double = 0
        var wB = 0
        var bestVariance: Double = -1
        var threshold = 128

        for i in 0..<256 {
            wB += histogram[i]
            if wB == 0 { continue }

            let wF = total - wB
            if wF == 0 { break }

            sumB += Double(i * histogram[i])
            let mB = sumB / Double(wB)
            let mF = (sum - sumB) / Double(wF)
            let between = Double(wB * wF) * (mB - mF) * (mB - mF)

            if between > bestVariance {
                bestVariance = between
                threshold = i
            }
        }

        return UInt8(threshold)
    }

    private func erode(mask: [UInt8], width: Int, height: Int) -> [UInt8] {
        var output = [UInt8](repeating: 0, count: mask.count)

        for y in 1..<(height - 1) {
            for x in 1..<(width - 1) {
                let i = y * width + x
                var keep = true
                for ny in (y - 1)...(y + 1) {
                    for nx in (x - 1)...(x + 1) {
                        if mask[ny * width + nx] == 0 {
                            keep = false
                            break
                        }
                    }
                    if !keep { break }
                }
                output[i] = keep ? 1 : 0
            }
        }

        return output
    }

    private func dilate(mask: [UInt8], width: Int, height: Int) -> [UInt8] {
        var output = [UInt8](repeating: 0, count: mask.count)

        for y in 1..<(height - 1) {
            for x in 1..<(width - 1) {
                let i = y * width + x
                var on = false
                for ny in (y - 1)...(y + 1) {
                    for nx in (x - 1)...(x + 1) {
                        if mask[ny * width + nx] == 1 {
                            on = true
                            break
                        }
                    }
                    if on { break }
                }
                output[i] = on ? 1 : 0
            }
        }

        return output
    }
}

private struct GrayscaleImage {
    let width: Int
    let height: Int
    let pixels: [UInt8]

    init?(pixelBuffer: CVPixelBuffer) {
        let format = CVPixelBufferGetPixelFormatType(pixelBuffer)
        guard format == kCVPixelFormatType_32BGRA else { return nil }

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }

        width = CVPixelBufferGetWidth(pixelBuffer)
        height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)

        var gray = [UInt8](repeating: 0, count: width * height)

        for y in 0..<height {
            let row = base.advanced(by: y * bytesPerRow).assumingMemoryBound(to: UInt8.self)
            let offset = y * width
            for x in 0..<width {
                let b = Int(row[x * 4 + 0])
                let g = Int(row[x * 4 + 1])
                let r = Int(row[x * 4 + 2])
                let yValue = (77 * r + 150 * g + 29 * b) >> 8
                gray[offset + x] = UInt8(clamping: yValue)
            }
        }

        pixels = gray
    }

    func crop(_ rect: IntRect) -> GrayscaleImage {
        var out = [UInt8](repeating: 0, count: rect.width * rect.height)

        for y in 0..<rect.height {
            let srcY = rect.y + y
            let srcOffset = srcY * width + rect.x
            let dstOffset = y * rect.width
            out[dstOffset..<(dstOffset + rect.width)] = pixels[srcOffset..<(srcOffset + rect.width)]
        }

        return GrayscaleImage(width: rect.width, height: rect.height, pixels: out)
    }

    private init(width: Int, height: Int, pixels: [UInt8]) {
        self.width = width
        self.height = height
        self.pixels = pixels
    }
}

private struct IntRect {
    let x: Int
    let y: Int
    let width: Int
    let height: Int
}

private struct Component {
    let indices: [Int]
    let count: Int
    let centroid: CGPoint
}

private struct SelectedComponent {
    let mask: [UInt8]
    let score: CGFloat
}

private struct Peak {
    let x: Int
    let y: Int
    let value: Int
}

private struct InstanceCluster {
    let count: Int
    let centroid: CGPoint
}

private extension Comparable {
    func clamped(to range: ClosedRange<Self>) -> Self {
        min(max(self, range.lowerBound), range.upperBound)
    }
}
