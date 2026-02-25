import CoreGraphics
import CoreImage
import Foundation
import Vision

struct TraySegmentationROI {
    let rect: CGRect
    let confidence: CGFloat
}

enum TrayContourSegmenter {
    static func detect(in image: CIImage, ciContext: CIContext, modelSide: CGFloat) -> TraySegmentationROI? {
        let modelRect = CGRect(x: 0, y: 0, width: modelSide, height: modelSide)
        guard let cgImage = ciContext.createCGImage(image, from: modelRect) else { return nil }

        let request = VNDetectContoursRequest()
        request.contrastAdjustment = 1.0
        request.detectsDarkOnLight = true
        request.maximumImageDimension = 512

        let handler = VNImageRequestHandler(cgImage: cgImage, orientation: .up, options: [:])
        do {
            try handler.perform([request])
        } catch {
            return nil
        }

        guard let observation = request.results?.first else {
            return nil
        }

        let contours = collectContours(from: observation.topLevelContours)
        guard !contours.isEmpty else { return nil }

        var bestRect: CGRect?
        var bestScore = -CGFloat.greatestFiniteMagnitude

        for contour in contours {
            let normalizedBox = contour.normalizedPath.boundingBox
            if normalizedBox.isNull || normalizedBox.width <= 0 || normalizedBox.height <= 0 { continue }

            let rect = CGRect(
                x: normalizedBox.origin.x * modelSide,
                y: normalizedBox.origin.y * modelSide,
                width: normalizedBox.width * modelSide,
                height: normalizedBox.height * modelSide
            )

            if rect.width < modelSide * 0.30 || rect.height < modelSide * 0.30 { continue }

            let area = (rect.width * rect.height) / (modelSide * modelSide)
            if area < 0.20 || area > 0.97 { continue }

            let centerX = rect.midX / modelSide
            let centerY = rect.midY / modelSide
            let centerPenalty = abs(centerX - 0.5) + abs(centerY - 0.5)

            let aspect = rect.width / max(rect.height, 1)
            let aspectPenalty = abs(log(max(0.001, aspect))) * 0.55

            let contourPointBonus = min(1.0, CGFloat(contour.pointCount) / 260)
            let score = area * 2.3 + contourPointBonus * 0.8 - centerPenalty * 1.5 - aspectPenalty

            if score > bestScore {
                bestScore = score
                bestRect = rect
            }
        }

        guard let bestRect else { return nil }

        // Convert score to a 0...1 confidence-like value for downstream gating.
        let confidence = max(0, min(1, (bestScore + 1.8) / 2.8))
        return TraySegmentationROI(rect: bestRect.integral, confidence: confidence)
    }

    private static func collectContours(from topLevel: [VNContour]) -> [VNContour] {
        var all: [VNContour] = []
        all.reserveCapacity(64)

        func visit(_ contour: VNContour) {
            all.append(contour)
            let children = contour.childContours
            if !children.isEmpty {
                for child in children {
                    visit(child)
                }
            }
        }

        for contour in topLevel {
            visit(contour)
        }
        return all
    }
}
