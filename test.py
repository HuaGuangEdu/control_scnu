path = r"C:\Users\30215\AppData\Local\Programs\blockly-electron\resources"
path2 = r"C:\Users\30215\AppData\Local\Programs\blockly-electron"
import os
print(path.split('blockly-electron')[0])
print(path2.split('blockly-electron')[0])
print(os.path.join(path.split('blockly-electron')[0],'blockly-electron'))