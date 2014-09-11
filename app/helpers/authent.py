googauth={}
with open('./app/helpers/googauth.cfg','r') as f:
    lines=f.read().split('\n')
    for l in lines:
        googauth[l.split('=')[0]]=l.split('=')[1]
