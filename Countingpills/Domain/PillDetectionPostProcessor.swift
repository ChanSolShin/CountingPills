import CoreGraphics
import Foundation

struct PillInferenceResult {
    let count: Int
    let points: [CGPoint] // normalized 0...1
}

final class PillDetectionPostProcessor {
    private let modelSize: Float = 640
    private let preNmsTopK = 700
    private let maxPerVariant = 180
    private let maxFinalPoints = 220
    private let maxConfidenceLogitAbs: Float = 16
    private let diagnosticsTag = "[PillDebug]"

    func process(
        _ tensor: ModelOutputTensor,
        expectedVariants: Int,
        scoreThreshold: Float = 0.24,
        iouThreshold: Float = 0.50
    ) -> PillInferenceResult {
        guard let layout = inferLayout(shape: tensor.shape, valuesCount: tensor.values.count) else {
            debugLog("start shape=\(tensor.shape) values=\(tensor.values.count) layout=none")
            return PillInferenceResult(count: 0, points: [])
        }

        let availableBatch = max(1, tensor.values.count / max(1, layout.features * layout.boxes))
        let effectiveLayout = TensorLayout(
            batch: max(layout.batch, availableBatch),
            boxes: layout.boxes,
            features: layout.features,
            isChannelMajor: layout.isChannelMajor
        )
        let batchCount = max(1, min(expectedVariants, availableBatch))

        debugLog(
            "start shape=\(tensor.shape) values=\(tensor.values.count) " +
            "layout=features:\(layout.features) boxes:\(layout.boxes) channelMajor:\(layout.isChannelMajor) batch=\(batchCount)"
        )

        var perVariantDetections: [[Detection]] = []
        perVariantDetections.reserveCapacity(batchCount)

        for variant in 0..<batchCount {
            let decoded = decodeVariant(
                values: tensor.values,
                layout: effectiveLayout,
                batchIndex: variant,
                scoreThreshold: scoreThreshold,
                iouThreshold: iouThreshold
            )
            debugLog("variant[\(variant)] detections=\(decoded.count)")
            perVariantDetections.append(decoded)
        }

        let selectedVariants = selectReliableVariants(perVariantDetections)
        if selectedVariants.count != perVariantDetections.count {
            debugLog("variant-pruned from=\(perVariantDetections.count) to=\(selectedVariants.count)")
        }

        let merged = mergeByConsensus(selectedVariants)
        debugLog("consensus variants=\(selectedVariants.count) merged=\(merged.count)")

        let points = merged.map { cluster in
            CGPoint(
                x: CGFloat(cluster.centerX / modelSize).clamped(to: 0...1),
                y: CGFloat(cluster.centerY / modelSize).clamped(to: 0...1)
            )
        }

        return PillInferenceResult(count: points.count, points: points)
    }

    private func decodeVariant(
        values: [Float],
        layout: TensorLayout,
        batchIndex: Int,
        scoreThreshold: Float,
        iouThreshold: Float
    ) -> [Detection] {
        guard layout.features >= 5 else { return [] }

        let scale = inferCoordinateScale(values: values, layout: layout, batchIndex: batchIndex)
        var candidates: [Detection] = []
        candidates.reserveCapacity(min(layout.boxes, preNmsTopK))

        for box in 0..<layout.boxes {
            let cxRaw = value(values, layout: layout, batchIndex: batchIndex, box: box, feature: 0)
            let cyRaw = value(values, layout: layout, batchIndex: batchIndex, box: box, feature: 1)
            let wRaw = value(values, layout: layout, batchIndex: batchIndex, box: box, feature: 2)
            let hRaw = value(values, layout: layout, batchIndex: batchIndex, box: box, feature: 3)

            let score = decodeClassScore(
                values: values,
                layout: layout,
                batchIndex: batchIndex,
                box: box,
                classOffset: 4,
                classCount: layout.features - 4
            )
            if score <= 0 { continue }

            guard let detection = normalizeDetection(
                cxRaw: cxRaw,
                cyRaw: cyRaw,
                wRaw: wRaw,
                hRaw: hRaw,
                score: score,
                scale: scale
            ) else {
                continue
            }

            candidates.append(detection)
        }

        guard !candidates.isEmpty else { return [] }
        candidates.sort { $0.score > $1.score }

        let topCandidates = Array(candidates.prefix(preNmsTopK))
        let topScore = topCandidates.first?.score ?? 0
        let adaptiveThreshold = max(scoreThreshold, min(0.72, topScore * 0.62))

        var thresholded = topCandidates.filter { $0.score >= adaptiveThreshold }
        thresholded = removeWeakEdgeNoise(thresholded)

        let nmsed = nms(thresholded, iouThreshold: iouThreshold)
        let deduped = deduplicateByCenter(nmsed)
        return Array(deduped.prefix(maxPerVariant))
    }

    private func mergeByConsensus(_ perVariantDetections: [[Detection]]) -> [ConsensusCluster] {
        guard !perVariantDetections.isEmpty else { return [] }

        var flattened: [(variant: Int, detection: Detection)] = []
        for (variant, detections) in perVariantDetections.enumerated() {
            flattened.append(contentsOf: detections.map { (variant, $0) })
        }

        guard !flattened.isEmpty else { return [] }
        flattened.sort { $0.detection.score > $1.detection.score }

        var clusters: [MutableCluster] = []
        clusters.reserveCapacity(flattened.count / 2)

        for item in flattened {
            let det = item.detection
            var bestIndex: Int?
            var bestDistance2 = Float.greatestFiniteMagnitude

            for index in clusters.indices {
                let cluster = clusters[index]
                let dx = det.cx - cluster.centerX
                let dy = det.cy - cluster.centerY
                let distance2 = dx * dx + dy * dy

                let meanSide = max(6, (cluster.meanSide + det.meanSide) * 0.5)
                let radius = max(10, min(24, meanSide * 0.48))
                let radius2 = radius * radius

                if distance2 <= radius2, distance2 < bestDistance2 {
                    bestIndex = index
                    bestDistance2 = distance2
                }
            }

            if let bestIndex {
                clusters[bestIndex].add(det, variant: item.variant)
            } else {
                clusters.append(MutableCluster(seed: det, variant: item.variant))
            }
        }

        let perVariantCounts = perVariantDetections.map(\.count)
        let medianVariantCount = medianInt(perVariantCounts)

        var finalized = clusters.map { $0.finalized() }

        finalized = finalized.filter { cluster in
            let edgeMargin = modelSize * 0.04
            let nearEdge = cluster.centerX <= edgeMargin ||
                cluster.centerX >= (modelSize - edgeMargin) ||
                cluster.centerY <= edgeMargin ||
                cluster.centerY >= (modelSize - edgeMargin)

            if cluster.variantHits >= 2 {
                return true
            }

            if cluster.variantHits == 1 {
                if nearEdge { return false }
                if cluster.avgScore >= 0.95, cluster.meanSide >= 10 { return true }
            }

            return false
        }

        if finalized.count >= 70 {
            let topBand = modelSize * 0.18
            let topCount = finalized.reduce(into: 0) { partial, cluster in
                if cluster.centerY <= topBand {
                    partial += 1
                }
            }
            let topRatio = Float(topCount) / Float(finalized.count)

            if topRatio > 0.22 {
                finalized = finalized.filter { cluster in
                    if cluster.centerY > topBand {
                        return true
                    }
                    return cluster.variantHits >= 3 || cluster.avgScore >= 0.90
                }
            }
        }

        finalized = suppressTopStripeArtifacts(finalized)
        finalized = deduplicateConsensusClusters(finalized)
        finalized = applyCountGuard(finalized, medianVariantCount: medianVariantCount, variantCount: perVariantDetections.count)

        finalized.sort {
            if $0.variantHits != $1.variantHits {
                return $0.variantHits > $1.variantHits
            }
            if $0.avgScore != $1.avgScore {
                return $0.avgScore > $1.avgScore
            }
            return $0.centerY < $1.centerY
        }

        return Array(finalized.prefix(maxFinalPoints))
    }

    private func removeWeakEdgeNoise(_ detections: [Detection]) -> [Detection] {
        guard !detections.isEmpty else { return [] }

        let edgeMargin = modelSize * 0.03
        let topScore = detections.first?.score ?? 0
        let weakEdgeFloor = max(0.40, topScore * 0.58)
        let tinyFloor = max(0.42, topScore * 0.60)

        return detections.filter { det in
            let nearEdge =
                det.cx <= edgeMargin ||
                det.cx >= (modelSize - edgeMargin) ||
                det.cy <= edgeMargin ||
                det.cy >= (modelSize - edgeMargin)

            if nearEdge && det.score < weakEdgeFloor {
                return false
            }

            if det.area < 65, det.score < tinyFloor {
                return false
            }

            return true
        }
    }

    private func deduplicateByCenter(_ detections: [Detection]) -> [Detection] {
        guard detections.count > 1 else { return detections }

        let minSides = detections.map(\.meanSide).sorted()
        let medianSide = minSides[minSides.count / 2]
        let baseRadius = max(7, min(16, medianSide * 0.38))

        var kept: [Detection] = []
        kept.reserveCapacity(detections.count)

        for det in detections {
            let radius = max(baseRadius, det.meanSide * 0.34)
            let radius2 = radius * radius

            var duplicate = false
            for existing in kept {
                let dx = det.cx - existing.cx
                let dy = det.cy - existing.cy
                if dx * dx + dy * dy <= radius2 {
                    duplicate = true
                    break
                }
            }

            if !duplicate {
                kept.append(det)
            }
        }

        return kept
    }

    private func deduplicateConsensusClusters(_ clusters: [ConsensusCluster]) -> [ConsensusCluster] {
        guard clusters.count > 1 else { return clusters }

        let sorted = clusters.sorted {
            if $0.variantHits != $1.variantHits { return $0.variantHits > $1.variantHits }
            if $0.avgScore != $1.avgScore { return $0.avgScore > $1.avgScore }
            return $0.maxScore > $1.maxScore
        }

        var kept: [ConsensusCluster] = []
        kept.reserveCapacity(sorted.count)

        for cluster in sorted {
            let radius = max(10, min(22, cluster.meanSide * 0.52))
            let radius2 = radius * radius
            var duplicate = false

            for existing in kept {
                let dx = cluster.centerX - existing.centerX
                let dy = cluster.centerY - existing.centerY
                if dx * dx + dy * dy <= radius2 {
                    duplicate = true
                    break
                }
            }

            if !duplicate {
                kept.append(cluster)
            }
        }
        return kept
    }

    private func suppressTopStripeArtifacts(_ clusters: [ConsensusCluster]) -> [ConsensusCluster] {
        guard clusters.count >= 20 else { return clusters }

        let topBand = modelSize * 0.17
        let lowerBand = modelSize * 0.32
        let topClusters = clusters.filter { $0.centerY <= topBand }
        guard topClusters.count >= 10 else { return clusters }

        let minX = topClusters.map(\.centerX).min() ?? 0
        let maxX = topClusters.map(\.centerX).max() ?? 0
        let xCoverage = (maxX - minX) / modelSize
        let topRatio = Float(topClusters.count) / Float(max(1, clusters.count))

        let meanY = topClusters.reduce(Float(0)) { $0 + $1.centerY } / Float(topClusters.count)
        let yVariance = topClusters.reduce(Float(0)) { partial, cluster in
            let d = cluster.centerY - meanY
            return partial + d * d
        } / Float(topClusters.count)
        let yStd = sqrt(max(0, yVariance))

        let lowerBandCount = clusters.reduce(into: 0) { partial, cluster in
            if cluster.centerY > topBand, cluster.centerY <= lowerBand {
                partial += 1
            }
        }

        let looksLikeStripe =
            xCoverage >= 0.55 &&
            yStd <= modelSize * 0.034 &&
            topRatio >= 0.22 &&
            lowerBandCount <= Int(Float(topClusters.count) * 0.55)

        guard looksLikeStripe else { return clusters }

        let filtered = clusters.filter { cluster in
            guard cluster.centerY <= topBand else { return true }
            return cluster.variantHits >= 6 && cluster.avgScore >= 0.90
        }
        debugLog("stripe-suppressed removed=\(clusters.count - filtered.count)")
        return filtered
    }

    private func selectReliableVariants(_ variants: [[Detection]]) -> [[Detection]] {
        guard variants.count >= 4 else { return variants }

        let counts = variants.map(\.count)
        let med = Float(medianInt(counts))
        let upper = max(med * 1.85, med + 14)
        let lower = max(1, Int((med * 0.20).rounded(.down)))

        var kept: [[Detection]] = []
        kept.reserveCapacity(variants.count)

        for variant in variants {
            let c = variant.count
            if c >= lower, Float(c) <= upper {
                kept.append(variant)
            }
        }

        let minimum = max(3, variants.count / 2)
        return kept.count >= minimum ? kept : variants
    }

    private func applyCountGuard(
        _ clusters: [ConsensusCluster],
        medianVariantCount: Int,
        variantCount: Int
    ) -> [ConsensusCluster] {
        guard !clusters.isEmpty else { return clusters }
        guard medianVariantCount > 0 else { return clusters }

        let expected = Float(medianVariantCount)
        let upperSoft = expected * 1.35 + 6
        guard Float(clusters.count) > upperSoft else { return clusters }

        var filtered = clusters.filter { cluster in
            cluster.variantHits >= 3 || (cluster.variantHits >= 2 && cluster.avgScore >= 0.88)
        }

        if filtered.count >= 6, Float(filtered.count) > expected * 1.25 + 5 {
            filtered = filtered.filter { cluster in
                cluster.variantHits >= min(4, max(3, variantCount / 2)) || cluster.avgScore >= 0.92
            }
        }

        debugLog("count-guard expected=\(medianVariantCount) before=\(clusters.count) after=\(filtered.count)")
        return filtered.isEmpty ? clusters : filtered
    }

    private func medianInt(_ values: [Int]) -> Int {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        return sorted[sorted.count / 2]
    }

    private func nms(_ detections: [Detection], iouThreshold: Float) -> [Detection] {
        guard !detections.isEmpty else { return [] }

        var selected: [Detection] = []
        selected.reserveCapacity(min(detections.count, maxPerVariant))

        for detection in detections {
            var keep = true
            for kept in selected {
                if iou(detection.rect, kept.rect) > CGFloat(iouThreshold) {
                    keep = false
                    break
                }
            }
            if keep {
                selected.append(detection)
            }
            if selected.count >= maxPerVariant {
                break
            }
        }

        return selected
    }

    private func normalizeDetection(
        cxRaw: Float,
        cyRaw: Float,
        wRaw: Float,
        hRaw: Float,
        score: Float,
        scale: Float
    ) -> Detection? {
        let cx = cxRaw * scale
        let cy = cyRaw * scale
        let w = abs(wRaw * scale)
        let h = abs(hRaw * scale)

        guard cx.isFinite, cy.isFinite, w.isFinite, h.isFinite else { return nil }
        if w < 5 || h < 5 || w > modelSize * 0.56 || h > modelSize * 0.56 { return nil }

        let aspect = max(w / max(1, h), h / max(1, w))
        if aspect > 6.5 { return nil }

        let x1 = max(0, cx - w * 0.5)
        let y1 = max(0, cy - h * 0.5)
        let x2 = min(modelSize, cx + w * 0.5)
        let y2 = min(modelSize, cy + h * 0.5)

        let clampedW = x2 - x1
        let clampedH = y2 - y1
        if clampedW < 5 || clampedH < 5 { return nil }

        return Detection(
            rect: CGRect(x: CGFloat(x1), y: CGFloat(y1), width: CGFloat(clampedW), height: CGFloat(clampedH)),
            score: score,
            cx: x1 + clampedW * 0.5,
            cy: y1 + clampedH * 0.5,
            area: clampedW * clampedH,
            meanSide: (clampedW + clampedH) * 0.5
        )
    }

    private func decodeClassScore(
        values: [Float],
        layout: TensorLayout,
        batchIndex: Int,
        box: Int,
        classOffset: Int,
        classCount: Int
    ) -> Float {
        guard classCount > 0 else { return 0 }

        var best: Float = 0
        for cls in 0..<classCount {
            let raw = value(values, layout: layout, batchIndex: batchIndex, box: box, feature: classOffset + cls)
            let score = decodeConfidence(raw)
            if score > best {
                best = score
            }
        }
        return best
    }

    private func inferCoordinateScale(values: [Float], layout: TensorLayout, batchIndex: Int) -> Float {
        let sampleCount = min(layout.boxes, 1200)
        var maxAbs: Float = 0

        for box in 0..<sampleCount {
            let cx = abs(value(values, layout: layout, batchIndex: batchIndex, box: box, feature: 0))
            let cy = abs(value(values, layout: layout, batchIndex: batchIndex, box: box, feature: 1))
            let w = abs(value(values, layout: layout, batchIndex: batchIndex, box: box, feature: 2))
            let h = abs(value(values, layout: layout, batchIndex: batchIndex, box: box, feature: 3))
            maxAbs = max(maxAbs, cx, cy, w, h)
        }

        return maxAbs <= 2.5 ? modelSize : 1
    }

    private func decodeConfidence(_ raw: Float) -> Float {
        if raw.isNaN || !raw.isFinite { return 0 }
        if raw >= 0, raw <= 1 { return raw }
        if raw > maxConfidenceLogitAbs { return 0.999 }
        if raw < -maxConfidenceLogitAbs { return 0.001 }
        return 1 / (1 + exp(-raw))
    }

    private func value(
        _ values: [Float],
        layout: TensorLayout,
        batchIndex: Int,
        box: Int,
        feature: Int
    ) -> Float {
        guard batchIndex >= 0, batchIndex < layout.batch else { return 0 }
        guard box >= 0, box < layout.boxes else { return 0 }
        guard feature >= 0, feature < layout.features else { return 0 }

        let batchStride = layout.features * layout.boxes
        let batchBase = batchIndex * batchStride

        let index: Int
        if layout.isChannelMajor {
            index = batchBase + feature * layout.boxes + box
        } else {
            index = batchBase + box * layout.features + feature
        }

        if index < 0 || index >= values.count { return 0 }
        return values[index]
    }

    private func inferLayout(shape: [Int], valuesCount: Int) -> TensorLayout? {
        let dims = shape.filter { $0 > 0 }

        if dims.count >= 3 {
            let batch = dims[dims.count - 3]
            let d1 = dims[dims.count - 2]
            let d2 = dims[dims.count - 1]

            if d1 >= 5, d1 <= 16, d2 >= 64 {
                let candidate = TensorLayout(batch: batch, boxes: d2, features: d1, isChannelMajor: true)
                if candidate.totalElements <= valuesCount {
                    return candidate
                }
            }

            if d2 >= 5, d2 <= 16, d1 >= 64 {
                let candidate = TensorLayout(batch: batch, boxes: d1, features: d2, isChannelMajor: false)
                if candidate.totalElements <= valuesCount {
                    return candidate
                }
            }
        }

        if dims.count == 2 {
            let d0 = dims[0]
            let d1 = dims[1]

            if d0 >= 5, d0 <= 16, d1 >= 64 {
                let candidate = TensorLayout(batch: 1, boxes: d1, features: d0, isChannelMajor: true)
                if candidate.totalElements <= valuesCount {
                    return candidate
                }
            }

            if d1 >= 5, d1 <= 16, d0 >= 64 {
                let candidate = TensorLayout(batch: 1, boxes: d0, features: d1, isChannelMajor: false)
                if candidate.totalElements <= valuesCount {
                    return candidate
                }
            }
        }

        if valuesCount % 6 == 0 {
            let boxes = valuesCount / 6
            if boxes >= 64 {
                return TensorLayout(batch: 1, boxes: boxes, features: 6, isChannelMajor: true)
            }
        }

        return nil
    }

    private func iou(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let inter = a.intersection(b)
        if inter.isNull || inter.width <= 0 || inter.height <= 0 { return 0 }

        let interArea = inter.width * inter.height
        let unionArea = a.width * a.height + b.width * b.height - interArea
        return unionArea > 0 ? interArea / unionArea : 0
    }

    private func debugLog(_ message: String) {
        #if DEBUG
        print("\(diagnosticsTag) \(message)")
        #endif
    }
}

private struct TensorLayout {
    let batch: Int
    let boxes: Int
    let features: Int
    let isChannelMajor: Bool

    var totalElements: Int {
        batch * boxes * features
    }
}

private struct Detection {
    let rect: CGRect
    let score: Float
    let cx: Float
    let cy: Float
    let area: Float
    let meanSide: Float
}

private struct MutableCluster {
    private(set) var sumX: Float
    private(set) var sumY: Float
    private(set) var weightedScoreSum: Float
    private(set) var weightSum: Float
    private(set) var meanSide: Float
    private(set) var maxScore: Float
    private(set) var count: Int
    private(set) var variantMask: UInt16

    var centerX: Float { weightSum > 0 ? sumX / weightSum : 0 }
    var centerY: Float { weightSum > 0 ? sumY / weightSum : 0 }

    init(seed: Detection, variant: Int) {
        let weight = max(0.2, seed.score)
        sumX = seed.cx * weight
        sumY = seed.cy * weight
        weightedScoreSum = seed.score * weight
        weightSum = weight
        meanSide = seed.meanSide
        maxScore = seed.score
        count = 1
        variantMask = 0
        if variant >= 0, variant < 16 {
            variantMask = UInt16(1 << UInt16(variant))
        }
    }

    mutating func add(_ detection: Detection, variant: Int) {
        let weight = max(0.2, detection.score)
        sumX += detection.cx * weight
        sumY += detection.cy * weight
        weightedScoreSum += detection.score * weight
        weightSum += weight
        meanSide = (meanSide * Float(count) + detection.meanSide) / Float(count + 1)
        maxScore = max(maxScore, detection.score)
        count += 1

        if variant >= 0, variant < 16 {
            variantMask |= UInt16(1 << UInt16(variant))
        }
    }

    func finalized() -> ConsensusCluster {
        let avgScore = weightSum > 0 ? (weightedScoreSum / weightSum) : 0
        return ConsensusCluster(
            centerX: centerX,
            centerY: centerY,
            avgScore: avgScore,
            maxScore: maxScore,
            meanSide: meanSide,
            pointCount: count,
            variantHits: Int(variantMask.nonzeroBitCount)
        )
    }
}

private struct ConsensusCluster {
    let centerX: Float
    let centerY: Float
    let avgScore: Float
    let maxScore: Float
    let meanSide: Float
    let pointCount: Int
    let variantHits: Int
}

private extension Comparable {
    func clamped(to range: ClosedRange<Self>) -> Self {
        min(max(self, range.lowerBound), range.upperBound)
    }
}
