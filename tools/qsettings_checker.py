from PySide6.QtCore import QCoreApplication, QSettings

app = QCoreApplication([])
app.setOrganizationName("WorthyLab")
app.setApplicationName("MonStim Analyzer")
s = QSettings()
print("Org:", s.organizationName())
print("App:", s.applicationName())
try:
    print("fileName:", s.fileName())
except Exception:
    print("fileName: (registry storage)")
print("Some keys:", s.allKeys()[:50])
