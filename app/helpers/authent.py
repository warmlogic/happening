googauth={}
with open('googauth.cfg','r') as f:
    lines=f.read().split('\n')
    for l in lines:
        googauth[l.split('=')[0]]=l.split('=')[1]
