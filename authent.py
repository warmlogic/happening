dbauth={}
with open('dbauth.cfg','r') as f:
    lines=f.read().split('\n')
    for l in lines:
        dbauth[l.split('=')[0]]=l.split('=')[1]
if 'port' in dbauth:
    dbauth['port']=int(dbauth['port'])
dbauth_m={}
with open('dbauth_m.cfg','r') as f:
    lines=f.read().split('\n')
    for l in lines:
        dbauth_m[l.split('=')[0]]=l.split('=')[1]
if 'port' in dbauth_m:
    dbauth_m['port']=int(dbauth_m['port'])
twitauth={}
with open('twitauth.cfg','r') as f:
    lines=f.read().split('\n')
    for l in lines:
        twitauth[l.split('=')[0]]=l.split('=')[1]
instaauth={}
with open('instaauth.cfg','r') as f:
    lines=f.read().split('\n')
    for l in lines:
        instaauth[l.split('=')[0]]=l.split('=')[1]
