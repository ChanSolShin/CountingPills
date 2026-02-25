import CoreGraphics
import CoreImage
import CoreML
import CoreVideo
import Foundation

enum TrayCoreMLSegmenter {
    private struct LoadedModel {
        let model: MLModel
        let inputName: String
    }

    private static let loadedModel: LoadedModel? = {
        guard let modelURL = Bundle.main.url(forResource: "tray_segmenter", withExtension: "mlmodelc") else {
            return nil
        }

        let config = MLModelConfiguration()
        config.computeUnits = .all

        guard
            let model = try? MLModel(contentsOf: modelURL, configuration: config),
            let firstInput = model.modelDescription.inputDescriptionsByName.first
        else {
            return nil
        }

        return LoadedModel(model: model, inputName: firstInput.key)
    }()

    static func detect(in image: CIImage, ciContext: CIContext, modelSide: CGFloat) -> TraySegmentationROI? {
        guard let loadedModel else { return nil }
        let side = Int(modelSide.rounded())
        guard side > 0 else { return nil }

        guard let pixelBuffer = makePixelBuffer(width: side, height: side) else { return nil }
        let source = image.cropped(to: CGRect(x: 0, y: 0, width: modelSide, height: modelSide))
        ciContext.render(source, to: pixelBuffer)

        guard
            let provider = try? MLDictionaryFeatureProvider(
                dictionary: [loadedModel.inputName: MLFeatureValue(pixelBuffer: pixelBuffer)]
            ),
            let prediction = try? loadedModel.model.prediction(from: provider),
            let mask = firstMultiArray(from: prediction),
            let maskRect = extractMaskRect(from: mask, modelSide: modelSide)
        else {
            return nil
        }

        return TraySegmentationROI(rect: maskRect.rect, confidence: maskRect.confidence)
    }

    private static func firstMultiArray(from provider: MLFeatureProvider) -> MLMultiArray? {
        for name in provider.featureNames {
            if let value = provider.featureValue(for: name)?.multiArrayValue {
                return value
            }
        }
        return nil
    }

    private static func extractMaskRect(
        from mask: MLMultiArray,
        modelSide: CGFloat
    ) -> (rect: CGRect, confidence: CGFloat)? {
        let dims = mask.shape.map(\.intValue)
        guard dims.count >= 2 else { return nil }

        guard let layout = inferLayout(dims: dims) else { return nil }
        let h = layout.height
        let w = layout.width
        guard h > 0, w > 0 else { return nil }

        let values = readFlatValues(mask)
        guard !values.isEmpty else { return nil }

        var minV = Float.greatestFiniteMagnitude
        var maxV = -Float.greatestFiniteMagnitude
        for value in values {
            minV = min(minV, value)
            maxV = max(maxV, value)
        }
        if !minV.isFinite || !maxV.isFinite || maxV <= minV { return nil }

        let normalizedThreshold: Float = maxV <= 1.5 ? 0.45 : 128

        var minX = w
        var minY = h
        var maxX = -1
        var maxY = -1
        var positiveCount = 0
        var positiveScore: Float = 0

        let total = w * h
        for y in 0..<h {
            for x in 0..<w {
                let value = valueAt(
                    x: x,
                    y: y,
                    dims: dims,
                    strides: mask.strides.map(\.intValue),
                    layout: layout,
                    values: values
                )

                if value >= normalizedThreshold {
                    positiveCount += 1
                    positiveScore += value
                    minX = min(minX, x)
                    minY = min(minY, y)
                    maxX = max(maxX, x)
                    maxY = max(maxY, y)
                }
            }
        }

        guard positiveCount > 0, maxX >= minX, maxY >= minY else { return nil }

        let areaRatio = Float(positiveCount) / Float(max(1, total))
        if areaRatio < 0.10 || areaRatio > 0.98 {
            return nil
        }

        let scaleX = modelSide / CGFloat(w)
        let scaleY = modelSide / CGFloat(h)
        let rect = CGRect(
            x: CGFloat(minX) * scaleX,
            y: CGFloat(minY) * scaleY,
            width: CGFloat(maxX - minX + 1) * scaleX,
            height: CGFloat(maxY - minY + 1) * scaleY
        ).integral

        let scoreNorm: CGFloat
        if maxV <= 1.5 {
            scoreNorm = CGFloat((positiveScore / Float(positiveCount)).clamped(to: 0...1))
        } else {
            scoreNorm = CGFloat((positiveScore / Float(positiveCount) / 255).clamped(to: 0...1))
        }
        let confidence = max(0, min(1, scoreNorm))
        return (rect, confidence)
    }

    private static func readFlatValues(_ array: MLMultiArray) -> [Float] {
        let count = array.count
        var output = [Float](repeating: 0, count: count)

        switch array.dataType {
        case .float32:
            let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: count)
            for i in 0..<count { output[i] = pointer[i] }
        case .double:
            let pointer = array.dataPointer.bindMemory(to: Double.self, capacity: count)
            for i in 0..<count { output[i] = Float(pointer[i]) }
        default:
            for i in 0..<count { output[i] = array[i].floatValue }
        }
        return output
    }

    private static func inferLayout(dims: [Int]) -> (height: Int, width: Int, hIndex: Int, wIndex: Int)? {
        if dims.count == 2 {
            return (dims[0], dims[1], 0, 1)
        }

        let indexed = dims.enumerated()
            .filter { $0.element > 8 }
            .sorted { $0.element > $1.element }

        guard indexed.count >= 2 else { return nil }
        let h = indexed[0]
        let w = indexed[1]
        return (h.element, w.element, h.offset, w.offset)
    }

    private static func valueAt(
        x: Int,
        y: Int,
        dims: [Int],
        strides: [Int],
        layout: (height: Int, width: Int, hIndex: Int, wIndex: Int),
        values: [Float]
    ) -> Float {
        var index = 0
        for dim in 0..<dims.count {
            let coordinate: Int
            if dim == layout.hIndex {
                coordinate = y
            } else if dim == layout.wIndex {
                coordinate = x
            } else {
                coordinate = 0
            }
            index += coordinate * strides[dim]
        }

        guard index >= 0, index < values.count else { return 0 }
        return values[index]
    }

    private static func makePixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
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

private extension Comparable {
    func clamped(to range: ClosedRange<Self>) -> Self {
        min(max(self, range.lowerBound), range.upperBound)
    }
}

