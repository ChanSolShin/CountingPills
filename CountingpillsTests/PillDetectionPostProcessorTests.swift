import CoreGraphics
import Testing
import UIKit
@testable import Countingpills

struct PillDetectionPostProcessorTests {
    @Test
    func postProcessorReturnsZeroForInvalidLayout() {
        let processor = PillDetectionPostProcessor()
        let tensor = ModelOutputTensor(values: [Float](repeating: 0, count: 10), shape: [1, 3, 3])

        let result = processor.process(tensor, expectedVariants: 1)

        #expect(result.count == 0)
        #expect(result.points.isEmpty)
    }

    @Test
    func postProcessorDetectsSingleCenteredPill() {
        let processor = PillDetectionPostProcessor()
        let shape = [1, 6, 64]
        var values = [Float](repeating: 0, count: 1 * 6 * 64)

        setValue(&values, shape: shape, batch: 0, feature: 0, box: 0, value: 0.50) // cx
        setValue(&values, shape: shape, batch: 0, feature: 1, box: 0, value: 0.50) // cy
        setValue(&values, shape: shape, batch: 0, feature: 2, box: 0, value: 0.09) // w
        setValue(&values, shape: shape, batch: 0, feature: 3, box: 0, value: 0.09) // h
        setValue(&values, shape: shape, batch: 0, feature: 4, box: 0, value: 0.10) // class0
        setValue(&values, shape: shape, batch: 0, feature: 5, box: 0, value: 0.98) // class1

        let tensor = ModelOutputTensor(values: values, shape: shape)
        let result = processor.process(tensor, expectedVariants: 1)

        #expect(result.count == 1)
        #expect(result.points.count == 1)
        if let point = result.points.first {
            #expect(abs(point.x - 0.5) < 0.02)
            #expect(abs(point.y - 0.5) < 0.02)
        }
    }

    @Test
    func postProcessorMergesSameDetectionAcrossVariants() {
        let processor = PillDetectionPostProcessor()
        let shape = [2, 6, 64]
        var values = [Float](repeating: 0, count: 2 * 6 * 64)

        for batch in 0..<2 {
            setValue(&values, shape: shape, batch: batch, feature: 0, box: 0, value: 0.51)
            setValue(&values, shape: shape, batch: batch, feature: 1, box: 0, value: 0.49)
            setValue(&values, shape: shape, batch: batch, feature: 2, box: 0, value: 0.08)
            setValue(&values, shape: shape, batch: batch, feature: 3, box: 0, value: 0.10)
            setValue(&values, shape: shape, batch: batch, feature: 4, box: 0, value: 0.05)
            setValue(&values, shape: shape, batch: batch, feature: 5, box: 0, value: 0.96)
        }

        let tensor = ModelOutputTensor(values: values, shape: shape)
        let result = processor.process(tensor, expectedVariants: 2)

        #expect(result.count == 1)
    }

    @Test
    func preparedFrameMappingKeepsCenterStable() {
        let frame = PreparedCapturedFrame(
            displayImage: UIImage(),
            modelPixelBuffers: [],
            inferenceROI: CGRect(x: 64, y: 64, width: 512, height: 512),
            edgeTrimRatio: 0.04,
            modelSide: 640
        )

        let mapped = frame.mapPointsToDisplay([CGPoint(x: 0.5, y: 0.5)])
        #expect(mapped.count == 1)

        if let center = mapped.first {
            #expect(abs(center.x - 0.5) < 0.001)
            #expect(abs(center.y - 0.5) < 0.001)
        }
    }

    @Test
    func preparedFrameMappingClampsToUnitSpace() {
        let frame = PreparedCapturedFrame(
            displayImage: UIImage(),
            modelPixelBuffers: [],
            inferenceROI: CGRect(x: -80, y: -80, width: 800, height: 800),
            edgeTrimRatio: 0.04,
            modelSide: 640
        )

        let mapped = frame.mapPointsToDisplay([CGPoint(x: 0, y: 0), CGPoint(x: 1, y: 1)])

        #expect(mapped.count == 2)
        for point in mapped {
            #expect(point.x >= 0 && point.x <= 1)
            #expect(point.y >= 0 && point.y <= 1)
        }
    }

    private func setValue(
        _ values: inout [Float],
        shape: [Int],
        batch: Int,
        feature: Int,
        box: Int,
        value: Float
    ) {
        let boxes = shape[2]
        let features = shape[1]
        let batchStride = features * boxes
        let index = batch * batchStride + feature * boxes + box
        values[index] = value
    }
}
