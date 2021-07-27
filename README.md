# Some simple steps


### Connecting to the cnaf.infn.it
Go to the the hpc-201-11-01-a machine.
I do it by 
```bash
ssh bologna
```

but you need to configure your `~/.ssh/config` by adding
```bash
Host bastion
 HostName bastion.cnaf.infn.it
 User sterbini
 ForwardX11 yes

Host bologna
 ProxyCommand ssh -q bastion nc hpc-201-11-01-a 22
 ForwardX11 yes
```

As you can see, `bologna` (hpc-201-11-01-a) host is passing via the `bastion` (bastion.cnaf.infn.it) connection.

### Activate the environment
I sugget sto open a `tmux` terminal and 

```bash
source /home/HPC/sterbini/py38/bin/activate
```
then move where you want to start lauch this DA study and make a clone of this repository

```
git clone https://github.com/sterbini/DA_study_example.git
cd DA_study_example
```












