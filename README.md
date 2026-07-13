<p align="center">
  <img src="https://github.com/user-attachments/assets/bcd564f6-48f6-4abb-b474-a81b3822c1a1" width="200"/>
  <img src="https://github.com/user-attachments/assets/9fb40ec3-14b2-4f75-836c-2f7cb2108894" width="200"/>
</p>

### 앱스토어 링크
https://apps.apple.com/us/app/countingpills-%EC%95%8C%EC%95%BD%EC%9D%84-%EB%B9%A0%EB%A5%B4%EA%B2%8C-%EC%84%B8%EC%96%B4%EB%B3%B4%EC%84%B8%EC%9A%94/id6759668362?l

### 앱 이름

- CountingPills

---

### 앱 소개

- 카메라로 알약 개수 카운팅

---

### 앱 목적

- 사용자가 카메라로 약을 촬영하면 AI 기반 이미지 분석으로 약 개수를 인식해 보여주는 앱

---

### 개발인원 
- 1명

### 사용기술

- **Language:** Swift
- **Framework:** SwiftUI
- **Architecture:** UseCase-based layered architecture
- **Asynchronous:** DispatchQueue-based concurrency
- **Version Control:** Git & GitHub
- **Model:** Roboflow instance segmentation model (pill_count-instance-segment/8)

---

### 주요 기능

1. **카메라 기반 약 촬영**
    
    사용자가 앱 내 카메라 화면에서 약을 프레임 안에 맞춰 촬영
    
2. **AI 기반 약 개수 인식**
    
    촬영된 이미지를 AI 모델이 분석해 화면 속 약의 개수를 자동으로 계산
    
3. **인식 결과 시각화**
    
    탐지된 약 위치를 화면 위에 포인트로 표시해, 어떤 약이 인식되었는지 직관적으로 확인
    
4. **촬영 결과 확인 및 재촬영**
    
    촬영 후 결과 화면에서 개수를 확인할 수 있고, 다시 촬영하기 버튼으로 즉시 재촬영
    
5. **온보딩 가이드 제공**
    
    앱 첫 실행 시 촬영 방법과 인식 정확도를 높이는 사용 가이드를 제공한다.
    
6. **다국어 로컬라이징 지원**
    
    사용자 기기 설정 언어에 따라 온보딩 가이드를 한국어 또는 영어로 자동 표시하도록 구현
   
---
## 아키텍처

CountingPills는 **SwiftUI/UIKit 하이브리드 기반의 UseCase 중심 계층형 아키텍처**로 구성되어 있습니다.

화면 UI는 SwiftUI로 구성하고, 카메라처럼 시스템 제어가 중요한 기능은 `UIViewControllerRepresentable`을 통해 UIKit으로 분리했습니다.

또한 촬영 프레임 전처리, AI 추론 실행, 외부 모델 호출, 카메라 세션 제어를 각각 별도 계층으로 나누어 UI와 비즈니스 로직, 시스템 제어 로직이 서로 섞이지 않도록 설계했습니다.

특히 카메라 세션은 별도의 `SessionController`에서 단일 serial queue로 제어하여 `AVCaptureSession`의 상태 전이 충돌을 방지했고, AI 추론은 UseCase와 Runner 계층으로 분리해 유지보수성과 확장성을 높였습니다.
```text
Countingpills/
├── Application
│   ├── CaptureFramePreparationUseCase.swift   # 촬영 프레임 전처리, ROI/리사이즈/모델 입력 변환
│   └── RunPillDetectionUseCase.swift          # 약 개수 추론 실행, Roboflow Runner 호출
│
├── Domain
│   └── PillDetectionTypes.swift               # 추론 결과, 포인트, 검출 정보 등 도메인 타입
│
├── Features
│   ├── Camera
│   │   ├── CameraCaptureScreen.swift          # SwiftUI 카메라 화면, 버튼/결과/로딩 UI
│   │   ├── CameraCaptureRepresentable.swift   # SwiftUI ↔ UIKit 브리지
│   │   ├── CameraCaptureViewController.swift  # 카메라 프리뷰, 프레임 수집, 추론 흐름 제어
│   │   ├── CameraCaptureStateMachine.swift    # 촬영/재촬영/처리중 상태 관리
│   │   └── CameraSessionController.swift      # AVCaptureSession 설정/시작/중지 단일 제어
│   │
│   └── Onboarding
│       └── OnboardingView.swift               # 첫 실행 가이드 화면
│
├── Settings.bundle
│   ├── Root.plist                             # 앱 설정 진입 및 오픈소스 라이선스 메뉴
│   ├── OpenSourceLicenses.plist               # 라이선스 표시 데이터
│   ├── ko.lproj                              # 설정 화면 한국어 리소스
│   └── en.lproj                              # 설정 화면 영어 리소스
│
├── en.lproj
│   └── Localizable.strings                    # 영어 로컬라이징 문자열
│
├── ko.lproj
│   └── Localizable.strings                    # 한국어 로컬라이징 문자열
│
├── PillRoboflowRunner.swift                   # Roboflow SDK 기반 모델 로드 및 추론 처리
├── ContentView.swift                          # 앱 메인 진입 화면, 온보딩/카메라 화면 라우팅
├── CountingpillsApp.swift                     # 앱 진입점
└── Info.plist                                 # 카메라 권한 및 앱 설정
```
---

## 배운점 및 성과

**배운점**

- AVCaptureSession은 단순히 카메라를 띄우는 수준이 아니라, 스레드와 상태 전이를 명확히 분리해 설계해야 안정적으로 동작한다는 점을 학습.
- AI 모델 연동에서는 모델 정확도뿐 아니라, 전처리 방식, 로딩 타이밍, 추론 결과 시각화까지 포함해 전체 파이프라인 관점에서 접근해야 한다는 점을 경험.
- 사용자 경험을 위해서는 기능 구현만이 아니라, 첫 실행 온보딩, 재촬영 흐름, 로딩 상태 안내처럼 사용자가 앱 상태를 이해할 수 있는 UX 설계가 중요하다는 점을 학습.
- 다국어 로컬라이징을 적용하면서 기능 완성도뿐 아니라 실제 배포 관점에서 사용자 환경을 고려한 개발이 필요하다는 점을 체감.

**성과**

- 카메라 촬영부터 AI 기반 약 개수 인식, 결과 시각화, 재촬영 흐름까지 하나의 사용자 시나리오를 직접 구현.
- AVCaptureSession 관련 랜덤 크래시를 분석하고, 세션 제어를 단일 흐름으로 정리해 카메라 기능의 안정성을 개선.
- AI 기반 인식 기능을 실제 iOS 앱에 연동하고, 촬영 결과를 사용자에게 직관적으로 보여주는 인터랙션까지 완성.
- 온보딩과 한국어/영어 로컬라이징을 적용해 실제 사용성과 배포 완성도를 높여 대한민국 다운로드 대비 5배 이상 해외 사용자의 다운로드 성과를 확인.

---
### 앱스토어 스크린샷

<p align="center">
  <img src="https://github.com/user-attachments/assets/190c2acd-9f89-42a2-89e7-188b5d780336" width="160"/>
  <img src="https://github.com/user-attachments/assets/5f7225e5-b83a-4f7f-9420-c225b624f872" width="160"/>
  <img src="https://github.com/user-attachments/assets/725392d0-5a8a-4243-92c9-183a41052f9b" width="160"/>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/8028aa40-f30a-4466-a78c-6d0f616224fb" width="160"/>
  <img src="https://github.com/user-attachments/assets/c8953bcc-7ca6-4f5c-8314-5119719c911b" width="160"/>
  <img src="https://github.com/user-attachments/assets/60dbf633-18f2-4f0b-a660-d5d263c1a812" width="160"/>
</p>


