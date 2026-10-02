There are two ways to control access, user/password authentication, and ip/domain checks. You can use either separately, or both.

### Turn on access control

The first thing you need to do to turn on access control is to set
the variable `access_control` in `your_db_dir/data/web_config.cfg` to `True`, such as:

```Python
access_control = True
```

You then need to configure user authentication, an IP/domain check, or both: with neither, nobody gets in.

Access control applies to the whole web app, its reports and scripts as well as its pages: a client that hasn't been let in
gets a 403 (Forbidden) for everything but the login screen.

### User authentication

To use user authentication, you need to create a logins.txt file inside your `your_db_dir/data/` directory. This can be a symlink.
If no file is found, no login succeeds. The login form sends the username and password in the body of a POST request, so
they don't end up in web server logs.
The logins.txt should one user/pass per line, separated by a tab, such as

```
username  password
another_user  another_password
```

### Domain and IP range check

To use this feature, you need to specify the location of the file in `web_config.cfg` in the `access_file` variable.

This file should contain 3 Python variables: `domain_list`, `allowed_ips`,
`blocked_ips`. Each variable should be a list containing the salient info.

The `domain_list` variable should be a list of domains allowed to access you database. They are matched against the host name
of the client's address, from a reverse DNS lookup: an entry lets in the names that contain it, so `uchicago.edu` lets in
`cs.uchicago.edu` (and `sc.edu` lets in `usc.edu`). The name isn't checked against the forward DNS, as many institutions'
proxies and VPNs have names that aren't: but whoever controls the reverse DNS of an address can give it a name in an allowed
domain, so the IP list is the stronger check.

```Python
domain_list = [
  "uchicago.edu",
  "indiana.edu",
  "louisiana.edu",
  "northwestern.edu"
]
```

The `allowed_ips` variable is a list of ips which are given access to the DB. Note that these are
matched using a regular expression, so you can express the whole ip, or just a part of it.

```Python
allowed_ips = [
  "128.135.",
  "128.32",
  "136.152",
  "136.153.1.1-255"
]
```

Note that the last IP notation expresses an IP range.

The `blocked_ips` variable is a list of IPs (exact matches needed) to deny access to:

```Python
blocked_ips = [
  "1.1.1.4"
]
```

### Behind a reverse proxy

The IP check uses the address of the client that connected to the web app, unless that is a reverse proxy: then it uses the
address the proxy appended to the `X-Forwarded-For` header (the last one there, as anything before it came from the client).
Connections to the web app's unix socket, and from `127.0.0.1` or `::1`, are taken to be from a proxy. If your proxy runs
elsewhere (another host, a Docker network), list its addresses in `/etc/philologic/philologic5.cfg`:

```Python
trusted_proxies = ["127.0.0.1", "::1", "172.17.0.1"]
```

### What happens when you're granted access

A cookie is saved to your browser, so that subsequent visits no longer require an access check. It lasts 7 days, and each
database has its own.

Cookies are signed with the database's key: the `secret` of its `data/db.locals.py`, if you set one, or else a key derived
for this database from the installation's secret, which `install.sh` writes to `/etc/philologic/philologic5.secret` (readable
by the web server's user only). Changing either one logs everybody out. Without the installation's secret, the web app draws
one each time it starts, so restarting it logs everybody out too.

The reports and scripts of a database with access control can't be read from web pages of other sites (they send no CORS
headers), as browsers of clients let in by their address could otherwise be used to read them.
