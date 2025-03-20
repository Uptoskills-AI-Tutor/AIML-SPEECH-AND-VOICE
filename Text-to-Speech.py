from gtts import gTTS
import os
# myText="Hello, I am jitendra behera from cuttack"

fh=open("test.txt","r")
myText =fh.read().replace("\n"," ")

language='en'

output=gTTS(text=myText,lang=language,slow=False)

output.save("output.mp3")
fh.close()
os.system("start output.mp3")