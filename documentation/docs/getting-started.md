# Getting Started

## Remote Access to the Desktop

```bash
ssh pucyril@pf-pc20.ethz.ch
```

## Mounting the Server (`pf/pfstudent/nimbus`)

The file server must be mounted via the desktop machine. This is done via SSHFS.

```bash
sudo sshfs -o allow_other,default_permissions pucyril@pf-pc20.ethz.ch:/home/pf/pfstud/nimbus /mnt/nimbus
```

## Improve SSH Performance

See IT Ticket...

```bash
sudo apt-get remove network-manager-config-connectivity-ubuntu
```
