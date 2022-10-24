# 作者：tomoya
# 创建：2022-09-29
# 更新：2022-09-29
# 用意：修改了一些pysound的源码并将其win端的代码放到这里供win的用户播放音频
import logging

logger = logging.getLogger(__name__)


class PlaysoundException(Exception):
    pass


def _canonicalizePath(path):
    """
    Support passing in a pathlib.Path-like object by converting to str.
    """
    import sys
    if sys.version_info[0] >= 3:
        return str(path)
    else:
        return path


def _playsoundWin(sound, block=True):
    """
    Utilizes windll.winmm. Tested and known to work with MP3 and WAVE on
    Windows 7 with Python 2.7. Probably works with more file formats.
    Probably works on Windows XP thru Windows 10. Probably works with all
    versions of Python.

    Inspired by (but not copied from) Michael Gundlach <gundlach@gmail.com>'s mp3play:
    https://github.com/michaelgundlach/mp3play

    I never would have tried using windll.winmm without seeing his code.
    """
    sound = _canonicalizePath(sound)
    if any((c in sound for c in ' "\'()')):
        from os import close, remove
        from os.path import splitext
        from shutil import copy
        from tempfile import mkstemp

        fd, tempPath = mkstemp(prefix='PS',
                               suffix=splitext(sound)[1])  # Avoid generating files longer than 8.3 characters.
        logger.info(
            'Made a temporary copy of {} at {} - use other filenames with only safe characters to avoid this.'.format(
                sound, tempPath))
        copy(sound, tempPath)
        close(fd)  # mkstemp opens the file, but it must be closed before MCI can open it.
        try:
            _playsoundWin(tempPath, block)
        finally:
            remove(tempPath)
        return

    from ctypes import c_buffer, windll
    from time import sleep

    def winCommand(*command):
        bufLen = 600
        buf = c_buffer(bufLen)
        command = ' '.join(command)  # .encode('utf-16')
        errorCode = int(
            windll.winmm.mciSendStringW(command, buf, bufLen - 1, 0))  # use widestring version of the function
        if errorCode:
            errorBuffer = c_buffer(bufLen)
            windll.winmm.mciGetErrorStringW(errorCode, errorBuffer,
                                            bufLen - 1)  # use widestring version of the function
            exceptionMessage = ('\n    Error ' + str(errorCode) + ' for command:'
                                                                  '\n        ' + command +
                                '\n    ' + errorBuffer.raw.decode('utf-16').rstrip('\0'))
            logger.error(exceptionMessage)
            raise PlaysoundException(exceptionMessage)
        return buf.value

    if '\\' in sound:
        sound = '"' + sound + '"'

    try:
        logger.debug('Starting')
        winCommand(u'open {}'.format(sound))
        winCommand(u'play {}{}'.format(sound, ' wait' if block else ''))
        logger.debug('Returning')
    finally:
        try:
            winCommand(u'close {}'.format(sound))
        except PlaysoundException:
            logger.warning(u'Failed to close the file: {}'.format(sound))
            pass


playsound = _playsoundWin
