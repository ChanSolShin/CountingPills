import Testing
@testable import Countingpills

struct CameraCaptureStateMachineTests {
    @Test
    func beginCaptureStartsProcessing() {
        var state = CameraCaptureStateMachine()

        #expect(state.isProcessing == false)
        #expect(state.isShowingCapturedFrame == false)
        #expect(state.beginCapture() == true)
        #expect(state.isProcessing == true)
    }

    @Test
    func beginCaptureFailsWhileProcessing() {
        var state = CameraCaptureStateMachine()
        #expect(state.beginCapture() == true)

        #expect(state.beginCapture() == false)
        #expect(state.isProcessing == true)
    }

    @Test
    func showingCapturedFrameBlocksNextCapture() {
        var state = CameraCaptureStateMachine()
        #expect(state.beginCapture() == true)
        state.markCapturedFrameShown()
        state.finishProcessing(keepCapturedFrame: true)

        #expect(state.isProcessing == false)
        #expect(state.isShowingCapturedFrame == true)
        #expect(state.beginCapture() == false)
    }

    @Test
    func resetForRetakeClearsCaptureState() {
        var state = CameraCaptureStateMachine()
        #expect(state.beginCapture() == true)
        state.markCapturedFrameShown()
        state.finishProcessing(keepCapturedFrame: true)
        state.resetForRetake()

        #expect(state.isProcessing == false)
        #expect(state.isShowingCapturedFrame == false)
        #expect(state.beginCapture() == true)
    }

    @Test
    func finishProcessingClearsCapturedFrameWhenRequested() {
        var state = CameraCaptureStateMachine()
        #expect(state.beginCapture() == true)
        state.markCapturedFrameShown()
        state.finishProcessing(keepCapturedFrame: false)

        #expect(state.isProcessing == false)
        #expect(state.isShowingCapturedFrame == false)
    }
}
