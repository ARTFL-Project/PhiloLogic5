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

To use this feature, you need to specify the location of the file in `web_config.cfg` in the `access_file` variable. It is a
[TOML](https://toml.io) file. A client is let in if any of its tables has a rule that matches it. Each table (its name is up
to you) can have a `domains` list and an `ip_rules` list; its other keys (`name`, `status`...) are ignored, so you can use
them to say whose rules they are. An optional `blocked_ips` list, which must come before the first table, names addresses to
refuse.

```toml
blocked_ips = ["192.0.2.99"]

[example-university]
name = "Example University"
domains = ["example.edu"]
ip_rules = [
    "192.0.2.",             # starts with
    "198.51.100.0/25",      # CIDR network
    "203.0.113.10-20",      # octet range
]
```

The rules of `ip_rules` are of three forms, and blank ones are ignored:

-   with a `/`, a CIDR network: `198.51.100.0/25` lets in `198.51.100.0` to `198.51.100.127`.
-   with a `-`, an octet range: each octet is a number or a range of numbers (`lo-hi`) between 0 and 255, and those that
    are left out can be anything. `203.0.113.10-20` lets in `203.0.113.10` to `203.0.113.20`; `a.b.c-d` lets in `a.b.c.0`
    to `a.b.d.255`, and `a.b-c.d.e` lets in `a.x.d.e` for each x from b to c. A rule that is not of this form
    (`192.0.2.4-192.0.2.9`, say) is skipped, with a warning in the web app's log.
-   any other rule is the start of the client's address as a string, dots included. `192.0.2.` lets in `192.0.2.x` only,
    whereas `192.0.2.1` lets in `192.0.2.1`, `192.0.2.10` to `192.0.2.19` and `192.0.2.100` to `192.0.2.199`.

The local networks (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16` and `127.0.0.0/8`) are always let in.

The entries of `blocked_ips` have to be the client's address exactly: `192.0.2.99` blocks that address only, and an entry that ends
with a dot blocks no one. A blocked address is refused whatever the other rules say.

The `domains` are matched against the host name of the client's address, from a reverse DNS lookup: an entry lets in the
names that contain it, so `uchicago.edu` lets in `cs.uchicago.edu` (and `sc.edu` lets in `usc.edu`). The name isn't checked
against the forward DNS, as many institutions' proxies and VPNs have names that aren't: but whoever controls the reverse DNS
of an address can give it a name in an allowed domain, so the IP rules are the stronger check. The lookup is only made for
clients that no IP rule lets in, and if the file has domains.

The web app reads the file again when its modification time changes, so edits take effect without a restart. A file that
can't be read, or isn't valid TOML (an access file written as Python, as it once was, say), lets nobody in: the web app's
log has an `ACCESS ERROR` line saying why.

### Behind a reverse proxy

The IP check uses the address of the client that connected to the web app, unless that is a reverse proxy: then it uses the
first address of the `X-Forwarded-For` header. Usually that is the only one, the address your proxy got the request from.
When a proxy or gateway before yours wrote its user's address there, it is that address, which nothing verifies; the web
app logs an `ACCESS AUDIT` line when the address your proxy saw connect would have been decided otherwise.
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
