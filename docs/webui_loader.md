# Loading databases from a web page: philologic5-webui-loader

`philologic5-webui-loader` is a web interface for loading PhiloLogic databases and editing their web config (`data/web_config.cfg`), without the command line. It runs `philoload5` for you, exactly as you would by hand, so everything it does can be done, or done again, from a shell.

With it you can:

-   pick the files to load, from a folder or a list of files, or upload them;
-   set the load options (those of `load_config.py`, see [Database Loading](database_loading.md)), with their help, starting from the defaults, from the options of an existing database, or from a load config file;
-   preview what the load will do on a sample of the files: the metadata found in the TEI headers, the tags of the texts and how they are treated, the words and sentences the parser finds, and the order of the documents;
-   check everything before launching the load (files, names, options, disk space, memory for a spaCy model...);
-   follow the load stage by stage, with its log, cancel it, and come back to it later: loads keep running when the page, the UI or the SSH connection is closed;
-   list the databases, load one again, or edit its web config (see [Configuring the Web Application](configure_web_app.md)).

It works in two ways:

1.  **On your own machine**, or on a server you connect to with SSH: you start it, it runs as you, and only you can use it.
2.  **As a service on a distant machine**: an administrator installs it once, and users open it in their browser, on the host of the databases (`https://<host>/philologic5-webui-loader/`), and log in, with no SSH or local installation needed.

## On your own machine

Run:

```
philologic5-webui-loader
```

It prints an address such as `http://localhost:8765/#/?token=...` and opens it in your browser. The token in the address is what gives access to the UI: keep it to yourself. The UI only answers on your own machine (127.0.0.1), and runs as you, so the databases you load belong to you, and you can edit the web config of the databases you can write.

This is also how it runs on a Mac. It links to the databases where PhiloLogic's gunicorn serves them itself, when it listens on a TCP address (`bind = "127.0.0.1:8080"` in `/var/lib/philologic5/web_app/gunicorn.conf.py`, as on a Mac): `http://127.0.0.1:8080/<name>/`. Behind a web server (gunicorn on a unix socket), it says where the web app serves them instead, `/<name>/` under the web server's URL prefix: there is no URL to set.

-   Stop it with Control-C. Loads it launched keep running.
-   `philologic5-webui-loader --background` keeps it running after the terminal closes; `philologic5-webui-loader url` prints its address again; `philologic5-webui-loader stop` stops it.
-   `--port` sets the first port tried (8765 by default, then the next free ones); `--no-browser` doesn't open the browser.

Its state (the directory of each load, with its command, load config, file list and log) is in `~/.local/state/philologic5/webui_loader`.

### On a server you connect to with SSH

Start it on the server in your SSH session: it then prints how to reach it from your own machine, with an SSH tunnel, such as:

```
ssh -N -L 8765:localhost:8765 you@server
```

Run this on your own machine and open the address printed by `philologic5-webui-loader` there. Next time, you can connect and start it in one go:

```
ssh -L 8765:localhost:8765 you@server philologic5-webui-loader
```

VS Code's Remote-SSH forwards the port by itself.

On a server shared by several users, the token is what keeps others from using your UI (and loading as you): don't share the address.

## The pages

**Databases** lists the databases of `database_root`: their title, number of documents, date of load and owner, whether they are being loaded or incomplete (a load which didn't finish), and whether you can edit their web config (if not, it says why: the files belong to someone else and aren't group-writable...). From there you can open a database, edit its web config, load it again with the same options, or start a new load from its options.

**New load** goes through four steps:

1.  *Files*: what to start from (the default options, those of an existing database, which saved them in `data/load_config.py`, or a load config file), the files, either on the computer hosting PhiloLogic (a folder with a pattern such as `*.xml`, possibly with its subfolders, or a file listing their paths, as `philoload5 -F`) or uploaded from your computer (see [Uploads](#uploads)), their format (TEI XML or plain text, TEI or Dublin Core headers, a bibliography), the name of the database and the number of cores to use.
2.  *Options*: the load options, grouped, with their help. Only the common ones are shown unless you ask for all of them; changed options are marked and can be reset to their default. Options which a load config sets with code (its own parser, load filters...) are shown but can't be changed here: its code is kept.
3.  *Previews*, on a sample of the files, with the options as they are: the document metadata the `doc_xpaths` find in the TEI headers (hover a value to see which XPath found it) and the order of the documents; the tags of the texts, how many, with which attributes, and how the load treats them (the object type they map to, not indexed, not breaking words); the words and sentences the parser finds in a file.
4.  *Review and launch*: the checks of the load (errors prevent launching it, warnings don't), the `philoload5` command it runs and the load config it writes, which only sets the options which differ from their default.

The load then has its page: its stages, the progress of the current one, its log (shown on request, and at once if the load failed), the files left out (no TEI header, invalid XML...) and, once done, the address of the database. **Loads** lists them all.

**Web config** edits `data/web_config.cfg`, grouped as in the file, with the metadata fields of the database to choose from. Each option has a form of its own: the citations (the named ones, which the lists of citations use by name, and those lists, in order), the aggregation and landing page settings, the sort orders, the replacements applied to queries and to the HTML of results (patterns and their replacements, in order, which you can try on a text: they are applied as the database applies them, and an invalid pattern can't be saved), and so on. "Edit as JSON" edits an option as JSON instead, for what its form can't show. Before saving, "Show changes" shows the lines which changed. Only the options you change are rewritten: the comments and the rest of the file stay as they are, and the previous version is kept as `web_config.cfg.<date>.bak`. Changes apply at once: reload the pages of the database to see them. `access_control` and `access_file` can't be changed from the UI.

## As a service on a distant machine

The service runs as a dedicated account, which runs the loads and owns the databases they create. Users have accounts of the UI itself, with a password and a second factor (an authenticator app): they don't need an account on the machine.

### Setting it up

`install.sh` sets the service up wherever systemd runs (as does `sudo extras/webui_loader/setup_service.sh`), unless the UI is turned off (see [Settings](#settings)). It:

-   creates the account of the service, `philologic` (`WEBUI_USER` sets another name), in the group which owns `database_root`, and gives `database_root` the setgid bit, so that the databases the service loads belong to the group and its members can still manage them from a shell (the service's umask is 002). The service can edit the web config of the databases it can write: those it loaded, and the group-writable ones;
-   adds the settings of the service to the global config, `/etc/philologic/philologic5.cfg` (only those it doesn't set yet: see [Settings](#settings)), installs the systemd unit and starts the service, on `127.0.0.1:8766`. Loads run apart from the service (`KillMode=process`): restarting it doesn't stop them. `install.sh` restarts it when it reinstalls PhiloLogic.
-   checks that your web server serves it (see [The web server](#web-server)), and if not, prints what to add to it.

Then:

1.  Have your web server serve the UI, once (see [The web server](#web-server)).
2.  Put the directories of your corpora in `webui_loader_allowed_roots`, in `/etc/philologic/philologic5.cfg`: users see nothing else of the machine (they can also upload files, in `webui_loader_upload_dir`, up to `webui_loader_upload_quota`). Restart the service: `sudo systemctl restart philologic5-webui-loader`.
3.  Create the first admin:

    ```
    sudo -u philologic philologic5-webui-loader users add yourname --admin
    ```

    This prints a temporary password. At the first login, the user sets up a second factor, scanning a QR code with an authenticator app (FreeOTP, Aegis, Google Authenticator, 1Password...), gets recovery codes (each lets them log in once without their device), and chooses their own password.

<a name="web-server"></a>

### The web server

The web server which serves the databases (Apache, Nginx...) also serves the UI, on their host and over HTTPS, at `https://<their host>/philologic5-webui-loader/`: it passes the requests of that path on to the service, as it does those of the databases to PhiloLogic's gunicorn. No other port, host name or certificate is needed, nor any URL setting: the UI takes its address from the requests, and links to the databases at `/philologic5/<name>/` on the same host, the URL prefix of PhiloLogic's install. With Apache (`sudo a2enmod proxy proxy_http headers`), add to the `<VirtualHost *:443>` of the databases:

```
<Location "/philologic5-webui-loader">
    ProxyPass "http://127.0.0.1:8766/philologic5-webui-loader" timeout=300
    ProxyPassReverse "http://127.0.0.1:8766/philologic5-webui-loader"
    RequestHeader set X-Forwarded-Proto "https"
</Location>
```

and reload Apache (`sudo systemctl reload apache2`). For Nginx, see `extras/webui_loader/nginx.conf` (it passes the host the browser asked for, which the service checks requests come from, and allows requests of 10 MB: upload chunks are up to 8 MB). The service refuses requests which don't come over HTTPS through the web server: it only trusts `X-Forwarded-Proto` and `X-Forwarded-For` from `webui_loader_forwarded_allow_ips` (`127.0.0.1`), and listens on `webui_loader_bind` (`127.0.0.1:8766`). Other users of the machine can reach that local port: the service trusts what they send as coming from the web server (still needing a login).

### Users and permissions

-   **Admins** manage the accounts, from the Users page or the command line, and can load or edit any database.
-   **Loaders** can load new databases, and load again or edit the web config of the databases they loaded with the service, or which an admin granted them. They only see their own loads. In web configs, they can't add HTML (which the pages of the databases show as such, in citations for instance), links other than `http://` and `https://`, or custom templates; they can keep what an admin set.

Accounts are managed with `philologic5-webui-loader users` (run as the account of the service):

```
philologic5-webui-loader users list
philologic5-webui-loader users add NAME [--admin]
philologic5-webui-loader users reset-password NAME
philologic5-webui-loader users reset-2fa NAME
philologic5-webui-loader users grant NAME DATABASE
philologic5-webui-loader users revoke NAME DATABASE
philologic5-webui-loader users admin|loader|disable|enable|remove NAME
```

The Users page does the same, and shows the audit log: logins and failed logins, loads, cancellations, web config edits, uploads and account changes, with who, when and from where.

<a name="uploads"></a>

### Uploads

Files can be uploaded from the computer of the browser to the one hosting PhiloLogic: on your own machine (useful when you reach it through an SSH tunnel), where they go in `~/.local/state/philologic5/webui_loader/uploads`, without limit, and in the service when `webui_loader_upload_dir` is set. Users can upload the files of a load: a zip or tar archive (the best way for thousands of files), or a folder. Uploads are sent in chunks: an upload which was interrupted goes on where it stopped when the same files are picked again. Each user has an upload area, limited by `webui_loader_upload_quota`; archives are extracted with checks (no files outside the upload, no links, their size counted as they are extracted).

<a name="settings"></a>

## Settings

The UI's settings are in PhiloLogic's global config, `/etc/philologic/philologic5.cfg`. It is read, not run: they must be literal values.

-   `webui_loader = False` turns the UI off on the machine (`install.sh` then stops and disables the service).
-   `webui_loader_allowed_roots`: the directories whose files can be loaded from the service (on your own machine, where the file browser starts).
-   `webui_loader_max_cores`: the most cores a load can use (all of them by default), for a machine shared with other work.
-   The settings of the service: `webui_loader_bind`, `webui_loader_forwarded_allow_ips`, `webui_loader_upload_dir`, `webui_loader_upload_quota`, `webui_loader_state_dir`, the session and login limits... `extras/webui_loader/settings.cfg` has them all, with their defaults and help.

Restart the service after changing them: `sudo systemctl restart philologic5-webui-loader`.

## Security

-   Load configs and web configs are Python files, which `philoload5` and the web app run. The UI never runs them: it reads their literal values, and only writes literal values, checking that what it wrote reads back as intended, with no other code. On your own machine, you can start a load from a load config with code of its own (its own parser...), which is kept, since you could run it anyway; the service refuses them, as it refuses to edit web configs with code, and the paths of files given in the options (lemma file, words to index, bibliography) must be in the allowed directories. Loads run with `python -P`, so that no directory of the user comes first in the path of Python modules.
-   File names with control characters (line breaks...) or spaces around them are refused, as philoload5 reads the list of files one per line.
-   Previews run in a process of their own, stopped after two minutes (a regular expression of the options can take forever).
-   On your own machine, the UI only answers requests addressed to localhost with its token (sent by the page from the browser's storage, which only its own address can read). In the service, sessions are kept by the server, their cookie is only sent over HTTPS to the UI's own path, and requests which change something must come from the UI's host (Origin) and carry the session's CSRF token.
-   The service shares the host of the databases, so scripts in their pages could use it as a logged-in user who views them: texts and web configs can contain such scripts (HTML in citations, attributes of TEI elements...). Give accounts to people you trust with the databases.
-   At the code step, users can check "Trust this browser": for `webui_loader_trusted_browser_days` (30; 0 turns it off), their password is then enough in that browser. Changing their password forgets their other trusted browsers; an admin resetting their password or second factor, or disabling their account, forgets them all; and they can forget them all from their account page.
-   Failed logins are counted per account and per address: after `webui_loader_max_failed_logins` (`webui_loader_max_failed_logins_per_ip`) within `webui_loader_lockout_time`, logins are refused for that time. Passwords are hashed with scrypt; the codes of the second factor can't be used twice.
-   The state of the service (`webui_loader_state_dir`: accounts, sessions, audit log, loads) is readable by its account only.
