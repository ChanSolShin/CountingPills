import CoreGraphics
import CoreImage
import CoreVideo
import UIKit

struct PreparedCapturedFrame {
    let displayImage: UIImage
    let modelPixelBuffers: [CVPixelBuffer]
    let inferenceROI: CGRect
    let edgeTrimRatio: CGFloat
    let modelSide: CGFloat

    func mapPointsToDisplay(_ points: [CGPoint]) -> [CGPoint] {
        guard modelSide > 0 else { return points }

        let trim = max(0, min(modelSide * 0.2, modelSide * edgeTrimRatio))
        let trimmedSide = max(1, modelSide - trim * 2)
        let trimScale = trimmedSide / modelSide

        return points.map { point in
            let xModel = point.x * modelSide
            let yModel = point.y * modelSide

            let xRoi = xModel * trimScale + trim
            let yRoi = yModel * trimScale + trim

            let xDisplay = inferenceROI.minX + (xRoi / modelSide) * inferenceROI.width
            let yDisplay = inferenceROI.minY + (yRoi / modelSide) * inferenceROI.height

            let nx = max(0, min(1, xDisplay / modelSide))
            let ny = max(0, min(1, yDisplay / modelSide))
            return CGPoint(x: nx, y: ny)
        }
    }
}

protocol CaptureFramePreparationUseCase {
    func prepare(from pixelBuffer: CVPixelBuffer) -> PreparedCapturedFrame?
}

final class DefaultCaptureFramePreparationUseCase: CaptureFramePreparationUseCase {
    private let modelSide: CGFloat
    private let modelRenderSize: CGSize
    private let ciContext: CIContext
    private let framePreprocessor: PillFramePreprocessor

    init(
        modelSide: CGFloat = 640,
        variantCount: Int = 8,
        ciContext: CIContext = CIContext()
    ) {
        self.modelSide = modelSide
        self.modelRenderSize = CGSize(width: modelSide, height: modelSide)
        self.ciContext = ciContext
        self.framePreprocessor = PillFramePreprocessor(modelSide: modelSide, variantCount: variantCount)
    }

    func prepare(from pixelBuffer: CVPixelBuffer) -> PreparedCapturedFrame? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let extent = ciImage.extent.integral
        let side = min(extent.width, extent.height)
        let cropRect = CGRect(
            x: extent.midX - side * 0.5,
            y: extent.midY - side * 0.5,
            width: side,
            height: side
        ).intersection(extent).integral

        guard !cropRect.isNull, cropRect.width > 0, cropRect.height > 0 else {
            return nil
        }

        let cropped = ciImage.cropped(to: cropRect)
        let translated = cropped.transformed(
            by: CGAffineTransform(
                translationX: -cropped.extent.origin.x,
                y: -cropped.extent.origin.y
            )
        )

        let sourceExtent = translated.extent.integral
        guard sourceExtent.width > 0, sourceExtent.height > 0 else { return nil }

        let modelScaleX = modelRenderSize.width / sourceExtent.width
        let modelScaleY = modelRenderSize.height / sourceExtent.height
        let resizedSquare = translated.transformed(by: CGAffineTransform(scaleX: modelScaleX, y: modelScaleY))
        let preprocessed = framePreprocessor.prepareForInference(from: resizedSquare, ciContext: ciContext)

        var modelPixelBuffers: [CVPixelBuffer] = []
        modelPixelBuffers.reserveCapacity(preprocessed.variantImages.count)
        for variantImage in preprocessed.variantImages {
            guard let buffer = makePixelBuffer(width: Int(modelRenderSize.width), height: Int(modelRenderSize.height)) else {
                return nil
            }
            ciContext.render(variantImage, to: buffer)
            modelPixelBuffers.append(buffer)
        }
        guard !modelPixelBuffers.isEmpty else { return nil }

        #if DEBUG
        let confidenceText: String
        if let confidence = preprocessed.roiConfidence {
            confidenceText = String(format: "%.2f", confidence)
        } else {
            confidenceText = "-"
        }

        if let roi = preprocessed.roiRect {
            print(
                "[PillDebug] preprocess-roi[\(preprocessed.roiSource)] x=\(Int(roi.origin.x)) y=\(Int(roi.origin.y)) " +
                "w=\(Int(roi.width)) h=\(Int(roi.height)) conf=\(confidenceText) variants=\(modelPixelBuffers.count)"
            )
        } else {
            print("[PillDebug] preprocess-roi[\(preprocessed.roiSource)] none conf=\(confidenceText) variants=\(modelPixelBuffers.count)")
        }
        #endif

        let displayRect = CGRect(origin: .zero, size: modelRenderSize)
        guard let displayCGImage = ciContext.createCGImage(resizedSquare, from: displayRect) else {
            return nil
        }

        return PreparedCapturedFrame(
            displayImage: UIImage(cgImage: displayCGImage),
            modelPixelBuffers: modelPixelBuffers,
            inferenceROI: preprocessed.roiRect ?? displayRect,
            edgeTrimRatio: preprocessed.edgeTrimRatio,
            modelSide: modelSide
        )
    }

    private func makePixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
        var pixelBuffer: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ]

        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary,
            &pixelBuffer
        )

        guard status == kCVReturnSuccess else { return nil }
        return pixelBuffer
    }
}
