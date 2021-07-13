'''
installation link:
https://www.videohelp.com/software/ffmpeg
'''

import os
import sys
os.system('ffmpeg -i {} -codec copy {}'.format(sys.argv[1], sys.argv[2]))