import Foundation

struct CameraCaptureStateMachine {
    private(set) var isProcessing = false
    private(set) var isShowingCapturedFrame = false

    mutating func beginCapture() -> Bool {
        guard !isProcessing, !isShowingCapturedFrame else { return false }
        isProcessing = true
        return true
    }

    mutating func markCapturedFrameShown() {
        isShowingCapturedFrame = true
    }

    mutating func finishProcessing(keepCapturedFrame: Bool) {
        isProcessing = false
        if !keepCapturedFrame {
            isShowingCapturedFrame = false
        }
    }

    mutating func resetForRetake() {
        isProcessing = false
        isShowingCapturedFrame = false
    }
}
