import CoreGraphics
import Foundation

struct PillInferenceResult {
    let count: Int
    let points: [CGPoint] // normalized 0...1
    let detections: [PillDetection]

    init(count: Int, points: [CGPoint], detections: [PillDetection] = []) {
        self.count = count
        self.points = points
        self.detections = detections
    }
}

struct PillDetection {
    let point: CGPoint // normalized 0...1
    let meanSide: CGFloat // in model-space pixels (640-base)
    let avgScore: Float
    let variantHits: Int
}
