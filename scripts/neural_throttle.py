"""Windows-only CPU limits for benchmark child processes, applied before startup."""
import ctypes
import os


def cpu_rate(percent, processors):
    """Windows uses hundredths of a percent of all logical processors."""
    if not 0 < percent <= 100 or not 1 <= processors <= 64:
        raise ValueError('CPU budget requires 0 < percent <= 100 and 1..64 processors')
    return max(1, round(percent * 100 / processors))


class WindowsCpuLimit:
    def __init__(self, percent):
        if os.name != 'nt':
            raise ValueError('CPU throttling is supported on Windows only')
        from ctypes import wintypes as w
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        signatures = {
            'CreateJobObjectW': ([ctypes.c_void_p, w.LPCWSTR], w.HANDLE),
            'SetInformationJobObject': ([w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD], w.BOOL),
            'QueryInformationJobObject': ([w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD, ctypes.c_void_p], w.BOOL),
            'AssignProcessToJobObject': ([w.HANDLE, w.HANDLE], w.BOOL),
            'OpenProcess': ([w.DWORD, w.BOOL, w.DWORD], w.HANDLE),
            'OpenThread': ([w.DWORD, w.BOOL, w.DWORD], w.HANDLE),
            'ResumeThread': ([w.HANDLE], w.DWORD),
            'CloseHandle': ([w.HANDLE], w.BOOL),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.kernel, name)
            function.argtypes, function.restype = arguments, result

        class Rate(ctypes.Structure):
            _fields_ = [('flags', w.DWORD), ('rate', w.DWORD)]

        self.rate_type = Rate
        self.rate = cpu_rate(percent, os.cpu_count())
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            info = Rate(0x1 | 0x4, self.rate)  # ENABLE | HARD_CAP
            self.check(self.kernel.SetInformationJobObject(self.handle, 15, ctypes.byref(info), ctypes.sizeof(info)))
            actual = Rate()
            self.check(self.kernel.QueryInformationJobObject(self.handle, 15, ctypes.byref(actual), ctypes.sizeof(actual), None))
            if actual.flags != info.flags or actual.rate != self.rate:
                raise RuntimeError('Windows did not retain the requested CPU cap')
        except BaseException:
            self.close()
            raise

    @staticmethod
    def check(result):
        if not result:
            raise ctypes.WinError(ctypes.get_last_error())

    def start(self, process, info, affinity):
        """Assign a CREATE_SUSPENDED process and resume its sole initial thread."""
        handle = self.kernel.OpenProcess(0x100 | 0x1, False, process.pid)
        self.check(handle)
        try:
            self.check(self.kernel.AssignProcessToJobObject(self.handle, handle))
            info.cpu_affinity(affinity)
            if sorted(info.cpu_affinity()) != sorted(affinity):
                raise RuntimeError('Windows did not retain the requested affinity')
            threads = info.threads()
            if len(threads) != 1:
                raise RuntimeError('Expected one suspended initial thread')
            thread = self.kernel.OpenThread(0x2, False, threads[0].id)
            self.check(thread)
            try:
                if self.kernel.ResumeThread(thread) == 0xffffffff:
                    raise ctypes.WinError(ctypes.get_last_error())
            finally:
                self.kernel.CloseHandle(thread)
        finally:
            self.kernel.CloseHandle(handle)

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def creation_flags(throttled):
    return 0x4 if throttled else 0  # CREATE_SUSPENDED, not exposed by subprocess
