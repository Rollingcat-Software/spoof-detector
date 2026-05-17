# amispoof subdomain migration runbook

**Goal:** move `https://fivucsas.com/amispoof/` → `https://amispoof.fivucsas.com/` without losing search rankings or breaking any existing inbound links.

**Constraint:** the `fivucsas.com` domain is **registered at TurkTicaret** but
**hosted on Hostinger** (Hetzner / shared). The DNS lives wherever the
nameservers point — check first.

---

## Step 0 — Find where DNS actually lives (5 min)

Where you create the new A record depends on which nameservers are
authoritative for `fivucsas.com`. Run this:

```bash
dig +short NS fivucsas.com
```

Three likely outcomes:

| Output | DNS lives at | Where to add the new A record |
|---|---|---|
| `ns1.turkticaret.net.` (or similar) | TurkTicaret panel | TurkTicaret → Domain Management → DNS |
| `ns1.dns-parking.com.` or Hostinger-style | Hostinger | Hostinger panel → Domains → DNS |
| Cloudflare nameservers | Cloudflare | Cloudflare dashboard → DNS |

The rest of this runbook assumes you found which one and have the panel open.

---

## Step 1 — Create the DNS A record (where DNS lives — Step 0)

Add an `A` record:

| Field | Value |
|---|---|
| Type | `A` |
| Name / Host | `amispoof` |
| Value / Target | `46.202.158.52` (Hostinger IP — same as the apex `fivucsas.com` A record) |
| TTL | `300` (5 min — short while migrating; bump to 3600 after stable) |

Wait 5–30 min for propagation. Verify:

```bash
dig +short amispoof.fivucsas.com
# expect: 46.202.158.52
```

If DNS lives at TurkTicaret but Hostinger is the host, this is the
only TurkTicaret-side step. Everything else happens on Hostinger.

---

## Step 2 — Create the subdomain on Hostinger (3 min)

1. Hostinger panel → **Domains** → `fivucsas.com` → **Subdomains**
2. Click **Create subdomain**
3. Subdomain: `amispoof`
4. Document root: `~/domains/fivucsas.com/public_html/amispoof_sub/`
   (a *new* folder; do NOT point it at the existing `public_html/amispoof/`)
5. Submit

Hostinger should auto-issue a Let's Encrypt SSL cert within ~5 min.
Verify: `https://amispoof.fivucsas.com` returns a TLS handshake (the page
will be empty / default-index for now — that's expected).

---

## Step 3 — First deploy to the new subdomain (2 min)

From the dev machine:

```bash
cd /opt/projects/fivucsas/spoof-detector/web
npm run build && npm run amispoof:bundle

# Push to the NEW subdomain root.
scp -P 65002 amispoof/index.html amispoof/app.js \
  u349700627@46.202.158.52:~/domains/fivucsas.com/public_html/amispoof_sub/

scp -P 65002 amispoof/lib/spoof-detector.js amispoof/lib/spoof-detector.js.map \
  amispoof/lib/spoof-detector-*.js amispoof/lib/spoof-detector-*.js.map \
  u349700627@46.202.158.52:~/domains/fivucsas.com/public_html/amispoof_sub/lib/

# The MediaPipe + ONNX models are required by app.js but Hostinger
# doesn't auto-create the `models/` folder — scp them too.
scp -P 65002 amispoof/models/minifasnet_v2.onnx amispoof/models/face_landmarker.task \
  u349700627@46.202.158.52:~/domains/fivucsas.com/public_html/amispoof_sub/models/
```

Verify: open `https://amispoof.fivucsas.com` on a mobile device,
allow camera, confirm the analyzer panel populates. Same page, new URL.

---

## Step 4 — Flip canonical + Open Graph URLs (this PR)

Once the new URL is serving correctly, change these in `web/amispoof/index.html`:

```diff
- <link rel="canonical" href="https://fivucsas.com/amispoof/" />
+ <link rel="canonical" href="https://amispoof.fivucsas.com/" />

- <meta property="og:url" content="https://fivucsas.com/amispoof/" />
+ <meta property="og:url" content="https://amispoof.fivucsas.com/" />
```

And in the JSON-LD `SoftwareApplication` block (search for the existing
`"url": "https://fivucsas.com/amispoof/"` line):

```diff
-     "url": "https://fivucsas.com/amispoof/",
+     "url": "https://amispoof.fivucsas.com/",
```

Redeploy to BOTH `amispoof_sub/` AND `public_html/amispoof/` (old location
still serves while Google migrates indexing).

---

## Step 5 — 301 redirect from the OLD URL

Add a `.htaccess` at `~/domains/fivucsas.com/public_html/amispoof/.htaccess`
on Hostinger with this content (file is also kept in this repo at
`SUBDOMAIN_HTACCESS_TEMPLATE.txt` for reference):

```apache
# Permanent redirect to the new subdomain. Inbound links and Google
# crawls hitting the old /amispoof/ slug will follow this 301 and
# Google will transfer ranking authority to the new URL over ~2-4 weeks.
RewriteEngine On
RewriteRule ^(.*)$ https://amispoof.fivucsas.com/$1 [R=301,L]
```

Verify:

```bash
curl -sI https://fivucsas.com/amispoof/
# expect: HTTP/2 301
# expect: location: https://amispoof.fivucsas.com/
```

---

## Step 6 — Google Search Console

1. Open <https://search.google.com/search-console/>
2. **Add property** → URL prefix → `https://amispoof.fivucsas.com/`
3. Verify ownership (DNS TXT record or HTML file upload — Hostinger
   makes both easy)
4. **Sitemaps** → submit `https://amispoof.fivucsas.com/sitemap.xml`
   (you'll need to create this — see Step 7)
5. **URL Inspection** → `https://amispoof.fivucsas.com/` → **Request indexing**
6. On the **OLD** `fivucsas.com` property, **URL Inspection** →
   `https://fivucsas.com/amispoof/` → **Request indexing** so Google
   sees the new 301 quickly

Migration window: Google starts honouring the 301 within days, full
ranking transfer takes 2–4 weeks.

---

## Step 7 — Optional sitemap (5 min)

Create `~/domains/fivucsas.com/public_html/amispoof_sub/sitemap.xml`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://amispoof.fivucsas.com/</loc>
    <lastmod>2026-05-17</lastmod>
    <changefreq>weekly</changefreq>
    <priority>1.0</priority>
  </url>
</urlset>
```

And a `robots.txt` if you don't already have one:

```
User-agent: *
Allow: /
Sitemap: https://amispoof.fivucsas.com/sitemap.xml
```

---

## Step 8 — Update the dev-side deploy runbook

After migration is verified, update `CLAUDE.md` in the parent repo:

```diff
- scp -P 65002 amispoof/index.html amispoof/app.js u349700627@46.202.158.52:~/domains/fivucsas.com/public_html/amispoof/
+ scp -P 65002 amispoof/index.html amispoof/app.js u349700627@46.202.158.52:~/domains/fivucsas.com/public_html/amispoof_sub/

- scp -P 65002 amispoof/lib/spoof-detector.js amispoof/lib/spoof-detector.js.map u349700627@46.202.158.52:~/domains/fivucsas.com/public_html/amispoof/lib/
+ scp -P 65002 amispoof/lib/spoof-detector.js amispoof/lib/spoof-detector.js.map u349700627@46.202.158.52:~/domains/fivucsas.com/public_html/amispoof_sub/lib/
```

Keep the OLD deploy target alive for ~30 days as well in case rollback
is needed. After that, simplify to single-target.

---

## Rollback plan

If anything goes wrong:

1. Remove the `.htaccess` redirect from Hostinger panel
2. Old `fivucsas.com/amispoof/` resumes serving normally (it has never
   stopped — the redirect was the only thing routing traffic away)
3. In Search Console, leave both properties; Google honours the most
   recent canonical hint

The migration is **fully reversible** until the old `public_html/amispoof/`
folder is deleted (which Step 8 explicitly does NOT do).

---

## When to delete the old `public_html/amispoof/` folder

After Search Console's old property shows **0 indexed pages from /amispoof/
in the last 28 days** (Indexing → Pages report). Typically 6–12 weeks
post-migration. Once that's clean:

```bash
ssh u349700627@46.202.158.52
rm -rf ~/domains/fivucsas.com/public_html/amispoof/
# also remove the .htaccess if it lived there
```

And drop the old deploy line from `CLAUDE.md`.
