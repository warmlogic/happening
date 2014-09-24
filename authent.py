dbauth={}
with open('dbauth.cfg','r') as f:
    lines=f.read().split('\n')
    for l in lines:
        dbauth[l.split('=')[0]]=l.split('=')[1]
if 'port' in dbauth:
    dbauth['port']=int(dbauth['port'])
dbauth_a={}
with open('dbauth_a.cfg','r') as f:
    lines=f.read().split('\n')
    for l in lines:
        dbauth_a[l.split('=')[0]]=l.split('=')[1]
if 'port' in dbauth_a:
    dbauth_a['port']=int(dbauth_a['port'])
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
