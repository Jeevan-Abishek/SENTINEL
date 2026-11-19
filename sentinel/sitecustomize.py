import platform

# Disable WMI in Python 3.14 to prevent Windows fatal exception 0xc0000008 in sandbox
try:
    platform._wmi = None
    def _mock_wmi_query(*a, **k):
        raise OSError("not supported")
    platform._wmi_query = _mock_wmi_query
except Exception:
    pass
