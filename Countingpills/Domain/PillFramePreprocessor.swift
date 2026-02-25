import CoreGraphics
import CoreImage
import Foundation

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
    private let allowContourFallback: Bool
    private let useMLTraySegmentation: Bool

    init(
        modelSide: CGFloat = 640,
        variantCount: Int = 6,
        edgeTrimRatio: CGFloat = 0.0,
        roiMode: ROIMode = .fixed640,
        allowContourFallback: Bool = true,
        useMLTraySegmentation: Bool = true
    ) {
        self.modelSide = modelSide
        self.variantCount = max(1, variantCount)
        self.edgeTrimRatio = max(0, min(0.2, edgeTrimRatio))
        self.roiMode = roiMode
        self.allowContourFallback = allowContourFallback
        self.useMLTraySegmentation = useMLTraySegmentation
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
        if useMLTraySegmentation,
           let segmented = TrayCoreMLSegmenter.detect(in: image, ciContext: ciContext, modelSide: modelSide)
        {
            let stabilized = stabilizeSquare(segmented.rect, in: modelRect)
            guard !stabilized.isNull, stabilized.width > 0, stabilized.height > 0 else { return nil }
            return DetectedROI(rect: stabilized, confidence: segmented.confidence)
        }

        guard allowContourFallback,
              let segmented = TrayContourSegmenter.detect(in: image, ciContext: ciContext, modelSide: modelSide)
        else {
            return nil
        }

        let stabilized = stabilizeSquare(segmented.rect, in: modelRect)
        guard !stabilized.isNull, stabilized.width > 0, stabilized.height > 0 else { return nil }
        return DetectedROI(rect: stabilized, confidence: segmented.confidence)
    }

    private func makeFinalROI(detected: DetectedROI?, fallback: CGRect, bounds: CGRect) -> FinalROI {
        guard let detected else {
            return FinalROI(rect: fallback, source: "fallback", confidence: nil)
        }
        guard detected.confidence >= 0.40 else {
            return FinalROI(rect: fallback, source: "fallback", confidence: detected.confidence)
        }
        let expanded = detected.rect.insetBy(dx: -modelSide * 0.04, dy: -modelSide * 0.04).intersection(bounds)
        let square = stabilizeSquare(expanded, in: bounds)
        guard !square.isNull, square.width >= modelSide * 0.74, square.height >= modelSide * 0.74 else {
            return FinalROI(rect: fallback, source: "fallback", confidence: detected.confidence)
        }
        return FinalROI(rect: square.integral, source: "detected", confidence: detected.confidence)
    }

    private func defaultROI(in extent: CGRect) -> CGRect {
        let side = min(extent.width, extent.height)
        let innerSide = side * 0.92
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
        // Keep inference-time preprocessing close to training distribution.
        // Aggressive contrast/clamp caused misses on bright white pills.
        return image
            .applyingFilter(
                "CINoiseReduction",
                parameters: [
                    "inputNoiseLevel": 0.02,
                    "inputSharpness": 0.20
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
                    kCIInputBrightnessKey: 0.02,
                    kCIInputContrastKey: 1.03
                ]
            )
        )
        push(
            base.applyingFilter(
                "CIColorControls",
                parameters: [
                    kCIInputSaturationKey: 1.0,
                    kCIInputBrightnessKey: -0.02,
                    kCIInputContrastKey: 1.03
                ]
            )
        )
        push(
            base.applyingFilter("CIExposureAdjust", parameters: [kCIInputEVKey: 0.08])
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
