import CoreGraphics
import CoreImage
import Foundation
import Vision

struct PillPreprocessedFrame {
    let primaryImage: CIImage
    let variantImages: [CIImage]
    let roiRect: CGRect?
    let roiSource: String
    let roiConfidence: CGFloat?
    let edgeTrimRatio: CGFloat
}

final class PillFramePreprocessor {
    enum ROIMode {
        case fixed640
        case detectedTray
    }

    private struct DetectedROI {
        let rect: CGRect
        let confidence: CGFloat
    }

    private struct FinalROI {
        let rect: CGRect
        let source: String
        let confidence: CGFloat?
    }

    private let modelSide: CGFloat
    private let variantCount: Int
    private let edgeTrimRatio: CGFloat
    private let roiMode: ROIMode

    init(
        modelSide: CGFloat = 640,
        variantCount: Int = 8,
        edgeTrimRatio: CGFloat = 0.0,
        roiMode: ROIMode = .fixed640
    ) {
        self.modelSide = modelSide
        self.variantCount = max(1, variantCount)
        self.edgeTrimRatio = max(0, min(0.2, edgeTrimRatio))
        self.roiMode = roiMode
    }

    func prepareForInference(from squareImage: CIImage, ciContext: CIContext) -> PillPreprocessedFrame {
        let normalized = normalizeToModelExtent(squareImage)

        let finalROI: FinalROI
        switch roiMode {
        case .fixed640:
            finalROI = FinalROI(rect: modelRect, source: "fixed", confidence: 1.0)
        case .detectedTray:
            let fallbackROI = defaultROI(in: normalized.extent)
            let detectedROI = detectTrayROI(in: normalized, ciContext: ciContext)
            finalROI = makeFinalROI(detected: detectedROI, fallback: fallbackROI, bounds: normalized.extent)
        }

        let cropped = cropAndResize(normalized, to: finalROI.rect)
        let trimmed = edgeTrimRatio > 0 ? trimEdgeBand(cropped) : cropped
        let enhanced = enhanceBaseImage(trimmed)
        let variants = makeVariants(from: enhanced)

        return PillPreprocessedFrame(
            primaryImage: variants.first ?? enhanced,
            variantImages: variants,
            roiRect: finalROI.rect,
            roiSource: finalROI.source,
            roiConfidence: finalROI.confidence,
            edgeTrimRatio: edgeTrimRatio
        )
    }

    private var modelRect: CGRect {
        CGRect(x: 0, y: 0, width: modelSide, height: modelSide)
    }

    private func normalizeToModelExtent(_ image: CIImage) -> CIImage {
        var normalized = image

        if normalized.extent.origin != .zero {
            normalized = normalized.transformed(
                by: CGAffineTransform(
                    translationX: -normalized.extent.origin.x,
                    y: -normalized.extent.origin.y
                )
            )
        }

        let extent = normalized.extent
        guard extent.width > 0, extent.height > 0 else {
            return normalized.cropped(to: modelRect)
        }

        if abs(extent.width - modelSide) < 0.5, abs(extent.height - modelSide) < 0.5 {
            return normalized.cropped(to: modelRect)
        }

        let scaleX = modelSide / extent.width
        let scaleY = modelSide / extent.height
        return normalized
            .transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
            .cropped(to: modelRect)
    }

    private func detectTrayROI(in image: CIImage, ciContext: CIContext) -> DetectedROI? {
        guard let cgImage = ciContext.createCGImage(image, from: modelRect) else { return nil }

        let request = VNDetectRectanglesRequest()
        request.maximumObservations = 10
        request.minimumSize = 0.30
        request.minimumAspectRatio = 0.72
        request.maximumAspectRatio = 1.45
        request.minimumConfidence = 0.55
        request.quadratureTolerance = 18

        let handler = VNImageRequestHandler(cgImage: cgImage, orientation: .up, options: [:])
        do {
            try handler.perform([request])
        } catch {
            return nil
        }

        guard let observations = request.results as? [VNRectangleObservation], !observations.isEmpty else {
            return nil
        }

        var bestRect: CGRect?
        var bestConfidence: CGFloat = 0
        var bestScore = -CGFloat.greatestFiniteMagnitude

        for observation in observations {
            let rect = CGRect(
                x: observation.boundingBox.origin.x * modelSide,
                y: observation.boundingBox.origin.y * modelSide,
                width: observation.boundingBox.width * modelSide,
                height: observation.boundingBox.height * modelSide
            )

            if rect.width < modelSide * 0.34 || rect.height < modelSide * 0.34 {
                continue
            }

            let area = (rect.width * rect.height) / (modelSide * modelSide)
            if area < 0.22 || area > 0.94 {
                continue
            }

            let centerX = rect.midX / modelSide
            let centerY = rect.midY / modelSide
            let centerPenalty = abs(centerX - 0.5) + abs(centerY - 0.5)
            let aspect = rect.width / max(rect.height, 1)
            let aspectPenalty = abs(log(max(0.001, aspect))) * 0.52
            let confidence = max(0, min(1, CGFloat(observation.confidence)))

            let score = area * 2.2 + confidence * 1.2 - centerPenalty * 1.6 - aspectPenalty
            if score > bestScore {
                bestScore = score
                bestRect = rect
                bestConfidence = confidence
            }
        }

        guard let bestRect else { return nil }
        let stabilized = stabilizeSquare(bestRect, in: modelRect)
        guard !stabilized.isNull, stabilized.width > 0, stabilized.height > 0 else { return nil }

        return DetectedROI(rect: stabilized, confidence: bestConfidence)
    }

    private func makeFinalROI(detected: DetectedROI?, fallback: CGRect, bounds: CGRect) -> FinalROI {
        guard let detected else {
            return FinalROI(rect: fallback, source: "fallback", confidence: nil)
        }
        guard detected.confidence >= 0.62 else {
            return FinalROI(rect: fallback, source: "fallback", confidence: detected.confidence)
        }

        let blend = 0.18 + detected.confidence * 0.20
        let originX = fallback.origin.x + (detected.rect.origin.x - fallback.origin.x) * blend
        let originY = fallback.origin.y + (detected.rect.origin.y - fallback.origin.y) * blend
        let width = fallback.width + (detected.rect.width - fallback.width) * blend
        let height = fallback.height + (detected.rect.height - fallback.height) * blend

        let blended = CGRect(x: originX, y: originY, width: width, height: height).intersection(bounds)
        let square = stabilizeSquare(blended, in: bounds)
        guard !square.isNull else {
            return FinalROI(rect: fallback, source: "fallback", confidence: detected.confidence)
        }

        let insetX = square.width * 0.08
        let insetY = square.height * 0.08
        let inner = square.insetBy(dx: insetX, dy: insetY).intersection(bounds)

        guard !inner.isNull, inner.width >= modelSide * 0.62, inner.height >= modelSide * 0.62 else {
            return FinalROI(rect: fallback, source: "fallback", confidence: detected.confidence)
        }
        return FinalROI(rect: inner.integral, source: "detected", confidence: detected.confidence)
    }

    private func defaultROI(in extent: CGRect) -> CGRect {
        let side = min(extent.width, extent.height)
        let innerSide = side * 0.86
        let originX = extent.midX - innerSide * 0.5
        let originY = extent.midY - innerSide * 0.5

        return CGRect(x: originX, y: originY, width: innerSide, height: innerSide)
            .intersection(extent)
            .integral
    }

    private func cropAndResize(_ image: CIImage, to roiRect: CGRect) -> CIImage {
        let cropped = image.cropped(to: roiRect).transformed(
            by: CGAffineTransform(translationX: -roiRect.origin.x, y: -roiRect.origin.y)
        )

        let scaleX = modelSide / roiRect.width
        let scaleY = modelSide / roiRect.height
        return cropped
            .transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
            .cropped(to: modelRect)
    }

    private func trimEdgeBand(_ image: CIImage) -> CIImage {
        let trim = modelSide * edgeTrimRatio
        let trimmedRect = image.extent.insetBy(dx: trim, dy: trim)
        guard trimmedRect.width > modelSide * 0.52, trimmedRect.height > modelSide * 0.52 else {
            return image.cropped(to: modelRect)
        }

        let translated = image
            .cropped(to: trimmedRect)
            .transformed(by: CGAffineTransform(translationX: -trimmedRect.origin.x, y: -trimmedRect.origin.y))

        let scale = modelSide / trimmedRect.width
        return translated
            .transformed(by: CGAffineTransform(scaleX: scale, y: scale))
            .cropped(to: modelRect)
    }

    private func enhanceBaseImage(_ image: CIImage) -> CIImage {
        let denoised = image.applyingFilter(
            "CINoiseReduction",
            parameters: [
                "inputNoiseLevel": 0.03,
                "inputSharpness": 0.0
            ]
        )

        let median = denoised.applyingFilter("CIMedianFilter")

        return median
            .applyingFilter(
                "CIColorControls",
                parameters: [
                    kCIInputSaturationKey: 0.98,
                    kCIInputBrightnessKey: 0.01,
                    kCIInputContrastKey: 1.10
                ]
            )
            .applyingFilter(
                "CIHighlightShadowAdjust",
                parameters: [
                    "inputHighlightAmount": 0.52,
                    "inputShadowAmount": 0.05
                ]
            )
            .applyingFilter(
                "CIColorClamp",
                parameters: [
                    "inputMinComponents": CIVector(x: 0.03, y: 0.03, z: 0.03, w: 0),
                    "inputMaxComponents": CIVector(x: 0.97, y: 0.97, z: 0.97, w: 1)
                ]
            )
            .cropped(to: modelRect)
    }

    private func makeVariants(from base: CIImage) -> [CIImage] {
        var variants: [CIImage] = []
        variants.reserveCapacity(variantCount)

        func push(_ image: CIImage) {
            guard variants.count < variantCount else { return }
            variants.append(normalizeToModelExtent(image).cropped(to: modelRect))
        }

        push(base)
        push(
            base.applyingFilter(
                "CIColorControls",
                parameters: [
                    kCIInputSaturationKey: 1.0,
                    kCIInputBrightnessKey: 0.03,
                    kCIInputContrastKey: 1.08
                ]
            )
        )
        push(
            base.applyingFilter(
                "CIColorControls",
                parameters: [
                    kCIInputSaturationKey: 0.95,
                    kCIInputBrightnessKey: -0.02,
                    kCIInputContrastKey: 1.10
                ]
            )
        )
        push(
            base
                .applyingFilter("CIMedianFilter")
                .applyingFilter(
                    "CIColorControls",
                    parameters: [
                        kCIInputSaturationKey: 0.92,
                        kCIInputBrightnessKey: 0.01,
                        kCIInputContrastKey: 1.14
                    ]
                )
        )
        push(
            base
                .applyingFilter("CIGaussianBlur", parameters: [kCIInputRadiusKey: 0.65])
                .cropped(to: modelRect)
                .applyingFilter(
                    "CIColorControls",
                    parameters: [
                        kCIInputSaturationKey: 1.0,
                        kCIInputBrightnessKey: 0,
                        kCIInputContrastKey: 1.08
                    ]
                )
        )
        push(
            base
                .applyingFilter("CIExposureAdjust", parameters: [kCIInputEVKey: 0.15])
                .applyingFilter(
                    "CIColorControls",
                    parameters: [
                        kCIInputSaturationKey: 0.96,
                        kCIInputBrightnessKey: 0,
                        kCIInputContrastKey: 1.06
                    ]
                )
        )
        push(
            base
                .applyingFilter("CIExposureAdjust", parameters: [kCIInputEVKey: -0.10])
                .applyingFilter(
                    "CIColorControls",
                    parameters: [
                        kCIInputSaturationKey: 0.96,
                        kCIInputBrightnessKey: 0,
                        kCIInputContrastKey: 1.13
                    ]
                )
        )
        push(
            base
                .applyingFilter(
                    "CIUnsharpMask",
                    parameters: [
                        kCIInputRadiusKey: 1.0,
                        kCIInputIntensityKey: 0.65
                    ]
                )
                .applyingFilter(
                    "CIColorControls",
                    parameters: [
                        kCIInputSaturationKey: 0.98,
                        kCIInputBrightnessKey: 0.01,
                        kCIInputContrastKey: 1.06
                    ]
                )
        )

        while variants.count < variantCount {
            variants.append(base)
        }

        return Array(variants.prefix(variantCount))
    }

    private func stabilizeSquare(_ rect: CGRect, in bounds: CGRect) -> CGRect {
        let clipped = rect.intersection(bounds)
        guard !clipped.isNull, clipped.width > 0, clipped.height > 0 else { return .null }

        let side = min(clipped.width, clipped.height)
        let x = clipped.midX - side * 0.5
        let y = clipped.midY - side * 0.5

        return CGRect(x: x, y: y, width: side, height: side)
            .intersection(bounds)
            .integral
    }
}
