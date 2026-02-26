import SwiftUI

struct OnboardingView: View {
    private struct Page: Identifiable {
        let id = UUID()
        let imageName: String
        let titleKey: String
        let bodyKey: String
    }

    private let pages: [Page] = [
        Page(
            imageName: "viewfinder",
            titleKey: "onboarding.page1.title",
            bodyKey: "onboarding.page1.body"
        ),
        Page(
            imageName: "camera.circle",
            titleKey: "onboarding.page2.title",
            bodyKey: "onboarding.page2.body"
        ),
        Page(
            imageName: "exclamationmark.triangle",
            titleKey: "onboarding.page3.title",
            bodyKey: "onboarding.page3.body"
        )
    ]

    @State private var selection = 0
    let onFinish: () -> Void

    var body: some View {
        ZStack {
            Color(red: 12.0 / 255.0, green: 12.0 / 255.0, blue: 12.0 / 255.0)
                .ignoresSafeArea()

            VStack(spacing: 24) {
                HStack {
                    Spacer()
                    Button(localized("onboarding.skip")) {
                        onFinish()
                    }
                    .foregroundColor(.white.opacity(0.85))
                    .font(.system(size: 16, weight: .semibold))
                }
                .padding(.horizontal, 24)
                .padding(.top, 8)

                TabView(selection: $selection) {
                    ForEach(Array(pages.enumerated()), id: \.offset) { index, page in
                        VStack(spacing: 22) {
                            Image(systemName: page.imageName)
                                .resizable()
                                .scaledToFit()
                                .frame(width: 78, height: 78)
                                .foregroundColor(Color(red: 33.0 / 255.0, green: 194.0 / 255.0, blue: 177.0 / 255.0))

                            Text(localized(page.titleKey))
                                .font(.system(size: 28, weight: .bold))
                                .foregroundColor(.white)
                                .multilineTextAlignment(.center)

                            Text(localized(page.bodyKey))
                                .font(.system(size: 18, weight: .medium))
                                .foregroundColor(.white.opacity(0.92))
                                .multilineTextAlignment(.center)
                                .lineSpacing(4)
                                .padding(.horizontal, 32)
                        }
                        .tag(index)
                    }
                }
                .tabViewStyle(.page(indexDisplayMode: .always))

                Button(action: nextAction) {
                    Text(selection == pages.count - 1 ? localized("onboarding.start") : localized("onboarding.next"))
                        .font(.system(size: 18, weight: .bold))
                        .foregroundColor(.black)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 16)
                        .background(Color(red: 33.0 / 255.0, green: 194.0 / 255.0, blue: 177.0 / 255.0))
                        .cornerRadius(14)
                }
                .padding(.horizontal, 24)
                .padding(.bottom, 28)
            }
        }
    }

    private func nextAction() {
        if selection < pages.count - 1 {
            withAnimation {
                selection += 1
            }
        } else {
            onFinish()
        }
    }

    private func localized(_ key: String) -> String {
        NSLocalizedString(key, comment: "")
    }
}

