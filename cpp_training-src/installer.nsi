!define PRODUCT_NAME "MotionDetector"
!define PRODUCT_DISPLAY "Motion Detector"
!define PRODUCT_VERSION "2.0.0"
!define PRODUCT_PUBLISHER "CV Lab"

Name "${PRODUCT_DISPLAY}"
OutFile "..\MotionDetector_Setup.exe"
InstallDir "$PROGRAMFILES64\${PRODUCT_NAME}"
RequestExecutionLevel admin

SetCompressor lzma
Unicode true

!define MUI_ICON "app.ico"
!define MUI_UNICON "app.ico"

!include "MUI2.nsh"
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"

Section "Install"
    SetOutPath "$INSTDIR"

    File "MotionDetectorQt.exe"
    File "app.ico"
    File "qt.conf"

    SetOutPath "$INSTDIR\platforms"
    File "platforms\qwindows.dll"

    SetOutPath "$INSTDIR"
    File "*.dll"

    CreateShortCut "$DESKTOP\Motion Detector.lnk" "$INSTDIR\MotionDetectorQt.exe" "" "$INSTDIR\app.ico" "" SW_SHOWNORMAL

    CreateDirectory "$SMPROGRAMS\${PRODUCT_DISPLAY}"
    CreateShortCut "$SMPROGRAMS\${PRODUCT_DISPLAY}\Motion Detector.lnk" "$INSTDIR\MotionDetectorQt.exe" "" "$INSTDIR\app.ico"
    CreateShortCut "$SMPROGRAMS\${PRODUCT_DISPLAY}\Uninstall.lnk" "$INSTDIR\uninstall.exe"

    WriteUninstaller "$INSTDIR\uninstall.exe"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}" "DisplayName" "${PRODUCT_DISPLAY}"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}" "DisplayVersion" "${PRODUCT_VERSION}"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}" "Publisher" "${PRODUCT_PUBLISHER}"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}" "DisplayIcon" "$INSTDIR\app.ico"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}" "UninstallString" "$INSTDIR\uninstall.exe"
SectionEnd

Section "Uninstall"
    RMDir /r "$INSTDIR"
    Delete "$DESKTOP\Motion Detector.lnk"
    RMDir /r "$SMPROGRAMS\${PRODUCT_DISPLAY}"
    DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}"
SectionEnd
