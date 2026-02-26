import SwiftUI

struct CameraCaptureScreen: View {
    @State private var captureToken = 0
    @State private var retakeToken = 0
    @State private var pillCount = 0
    @State private var isProcessing = false
    @State private var isShowingCapturedFrame = false

    var body: some View {
        GeometryReader { proxy in
            ZStack {
                CameraCaptureRepresentable(
                    captureToken: captureToken,
                    retakeToken: retakeToken,
                    pillCount: $pillCount,
                    isProcessing: $isProcessing,
                    isShowingCapturedFrame: $isShowingCapturedFrame
                )
                .ignoresSafeArea()

                if isShowingCapturedFrame && !isProcessing {
                    let frameRect = modelFrameRect(in: proxy.size)
                    Text("\(pillCount)")
                        .font(.system(size: 36, weight: .bold))
                        .foregroundColor(Color(red: 33.0 / 255.0, green: 194.0 / 255.0, blue: 177.0 / 255.0))
                        .shadow(color: .black.opacity(0.75), radius: 4, x: 0, y: 2)
                        .position(x: frameRect.midX, y: countLabelY(frameRect: frameRect, canvasSize: proxy.size))
                }

                VStack {
                    Spacer()

                    if isShowingCapturedFrame && !isProcessing {
                        Button(action: {
                            pillCount = 0
                            retakeToken &+= 1
                        }) {
                            ZStack {
                                Circle()
                                    .fill(.white)
                                    .frame(width: 78, height: 78)

                                Circle()
                                    .stroke(.black.opacity(0.8), lineWidth: 2)
                                    .frame(width: 66, height: 66)

                                Image(systemName: "arrow.counterclockwise")
                                    .font(.system(size: 28, weight: .bold))
                                    .foregroundColor(.black)
                            }
                        }
                        .disabled(isProcessing)
                        .opacity(isProcessing ? 0.6 : 1.0)
                    } else {
                        Button(action: { captureToken &+= 1 }) {
                            ZStack {
                                Circle()
                                    .fill(.white)
                                    .frame(width: 78, height: 78)

                                Circle()
                                    .stroke(.black.opacity(0.8), lineWidth: 2)
                                    .frame(width: 66, height: 66)
                            }
                        }
                        .disabled(isProcessing || isShowingCapturedFrame)
                        .opacity((isProcessing || isShowingCapturedFrame) ? 0.6 : 1.0)
                    }
                }
                .padding(.bottom, 36)

                if isProcessing {
                    VStack(spacing: 8) {
                        ProgressView()
                    }
                    .padding(12)
                    .background(Color.black.opacity(0.7))
                    .cornerRadius(12)
                }
            }
        }
    }

    private func modelFrameRect(in size: CGSize) -> CGRect {
        let horizontalPadding: CGFloat = 16
        let topInset: CGFloat = 120
        let bottomReservedHeight: CGFloat = 280
        let targetSide: CGFloat = 352

        let availableWidth = max(1, size.width - (horizontalPadding * 2))
        let availableHeight = max(1, size.height - topInset - bottomReservedHeight)
        let side = min(targetSide, availableWidth, availableHeight)

        let x = (size.width - side) * 0.5
        let y = topInset + max(0, (availableHeight - side) * 0.5)
        return CGRect(x: x, y: y, width: side, height: side)
    }

    private func countLabelY(frameRect: CGRect, canvasSize: CGSize) -> CGFloat {
        let desiredY = frameRect.maxY + 34
        let maxYBeforeButtonArea = canvasSize.height - 170
        return min(desiredY, maxYBeforeButtonArea)
    }
}
