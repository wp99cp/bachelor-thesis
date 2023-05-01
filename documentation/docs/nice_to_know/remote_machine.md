## Mounting Remote Folders

To access the remote device (euler, pf-pc20, etc.) you can use the following commands to mount the corresponding
folders:

```bash
sshfs pucyril@euler.ethz.ch:/cluster/scratch/pucyril /mnt/euler
sshfs pucyril@pf-pc20.ethz.ch:/scratch2/pucyril /mnt/pf-pc20
```

### Mounting the Server (`pf/pfstudent/nimbus`)

The file server must be mounted via the desktop machine. This is done via SSHFS.

```bash
# Read-Only Access:
sshfs -o default_permissions \
  pucyril@pf-pc20.ethz.ch:/home/pf/pfstud/nimbus /mnt/nimbus

#Read-Write Access:
sshfs pucyril@pf-pc20.ethz.ch:/home/pf/pfstud/nimbus /mnt/nimbus
```