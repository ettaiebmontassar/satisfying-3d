"""Envoie une vidéo verticale sur YouTube (Short) via l'API YouTube Data v3.

Usage :
  python3 tools/upload_youtube.py tunnel_final.mp4 templates/tunnel/meta.json
  python3 tools/upload_youtube.py tunnel_final.mp4 templates/tunnel/meta.json --publish-at 2026-11-12T18:00+01:00
  python3 tools/upload_youtube.py tunnel_final.mp4 templates/tunnel/meta.json --privacy public
  python3 tools/upload_youtube.py tunnel_final.mp4 templates/tunnel/meta.json --dry-run

Par défaut la vidéo est envoyée en PRIVÉ : tu la vérifies dans YouTube Studio avant de la publier.
Première utilisation : place client_secret.json (identifiant OAuth « application de bureau »)
dans le dossier du kit ; une fenêtre de navigateur s'ouvre pour autoriser ta chaîne, puis le
jeton est gardé dans youtube_token.json (ne le partage jamais).
"""
import argparse, datetime as dt, json, os, random, subprocess, sys, time

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height:format=duration", "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    return int(d["streams"][0]["width"]), int(d["streams"][0]["height"]), float(d["format"]["duration"])


def build_body(meta, privacy, publish_at):
    title = meta["title"].strip()
    desc = meta.get("description", "").strip()
    if "#shorts" not in (title + " " + desc).lower():
        desc = (desc + "\n\n#Shorts").strip()
    if len(title) > 100:
        sys.exit(f"Titre trop long ({len(title)} caractères, maximum 100).")
    status = {
        "privacyStatus": privacy,
        "selfDeclaredMadeForKids": bool(meta.get("made_for_kids", False)),
    }
    if "synthetic" in meta:  # déclaration « contenu modifié ou synthétique »
        status["containsSyntheticMedia"] = bool(meta["synthetic"])
    if publish_at:
        status["privacyStatus"] = "private"  # exigé par l'API pour une publication programmée
        status["publishAt"] = publish_at
    return {
        "snippet": {
            "title": title,
            "description": desc,
            "tags": meta.get("tags", []),
            "categoryId": str(meta.get("category", "27")),  # 27 = Éducation
            "defaultLanguage": meta.get("language", "ar"),
            "defaultAudioLanguage": meta.get("language", "ar"),
        },
        "status": status,
    }


def to_rfc3339(s):
    d = dt.datetime.fromisoformat(s)
    if d.tzinfo is None:
        d = d.astimezone()  # fuseau local si non précisé
    d = d.astimezone(dt.timezone.utc)
    if d <= dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=5):
        sys.exit("La date de publication doit être dans le futur.")
    return d.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def credentials(secret, token):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    creds = Credentials.from_authorized_user_file(token, SCOPES) if os.path.exists(token) else None
    if creds and creds.valid:
        return creds
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        if os.environ.get("CI"):   # serveur sans navigateur : impossible de se connecter ici
            sys.exit("Jeton YouTube absent ou révoqué. Reconnecte-toi une fois sur ton PC "
                     "(python3 tools/upload_youtube.py ... --dry-run ne suffit pas : fais un vrai envoi), "
                     "puis remets le contenu de youtube_token.json dans le secret GitHub YT_TOKEN.")
        if not os.path.exists(secret):
            sys.exit(f"Fichier {secret} introuvable : crée un identifiant OAuth « application de bureau » "
                     "dans Google Cloud (voir LISEZMOI.md).")
        creds = InstalledAppFlow.from_client_secrets_file(secret, SCOPES).run_local_server(port=0)
    with open(token, "w") as f:
        f.write(creds.to_json())
    os.chmod(token, 0o600)
    return creds


def upload(youtube, path, body):
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload
    req = youtube.videos().insert(part="snippet,status", body=body,
                                  media_body=MediaFileUpload(path, chunksize=8 * 1024 * 1024, resumable=True))
    resp, tries = None, 0
    while resp is None:
        try:
            st, resp = req.next_chunk()
            if st:
                print(f"  envoi {int(st.progress() * 100)} %", flush=True)
        except HttpError as e:
            if e.resp.status in (500, 502, 503, 504) and tries < 8:
                tries += 1; time.sleep(min(60, 2 ** tries + random.random()))
            else:
                raise
    return resp


def external_thumb(meta_path):
    """Miniature illustrée déposée dans thumbnails/<modèle>/thumbnail.(png|jpg|jpeg|webp).
    Ramenée à 1280x720 en JPEG si elle dépasse 2 Mo ou n'est pas en 16:9 (limites YouTube)."""
    slug = os.path.basename(os.path.dirname(os.path.abspath(meta_path)))
    for ext in ("png", "jpg", "jpeg", "webp"):
        src = os.path.join(KIT, "thumbnails", slug, "thumbnail." + ext)
        if os.path.exists(src):
            break
    else:
        return None
    from PIL import Image
    im = Image.open(src)
    if os.path.getsize(src) <= 2_000_000 and ext in ("png", "jpg", "jpeg") and abs(im.width / im.height - 16 / 9) < 0.02:
        return src
    tw, th = 1280, 720
    s = max(tw / im.width, th / im.height)
    im = im.convert("RGB").resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    x0, y0 = (im.width - tw) // 2, (im.height - th) // 2
    out = os.path.join(os.path.dirname(src), "_upload.jpg")
    im.crop((x0, y0, x0 + tw, y0 + th)).save(out, quality=90)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video"); ap.add_argument("meta")
    ap.add_argument("--privacy", choices=["private", "unlisted", "public"], default="private")
    ap.add_argument("--publish-at", help="ex. 2026-11-12T18:00+01:00 (la vidéo reste privée jusque-là)")
    ap.add_argument("--thumbnail", help="image de miniature (facultatif)")
    ap.add_argument("--client-secret", default=os.path.join(KIT, "client_secret.json"))
    ap.add_argument("--token", default=os.path.join(KIT, "youtube_token.json"))
    ap.add_argument("--dry-run", action="store_true", help="vérifie tout sans rien envoyer")
    a = ap.parse_args()

    meta = json.load(open(a.meta, encoding="utf8"))
    w, h, dur = probe(a.video)
    if h <= w:
        print(f"ATTENTION : {w}x{h} n'est pas vertical, YouTube ne le traitera pas comme un Short.")
    if dur > 180:
        print(f"ATTENTION : {dur:.0f} s dépasse 3 minutes, ce ne sera pas un Short.")
    publish_at = to_rfc3339(a.publish_at or meta["publish_at"]) if (a.publish_at or meta.get("publish_at")) else None
    body = build_body(meta, a.privacy, publish_at)
    print(json.dumps(body, ensure_ascii=False, indent=2))
    print(f"Vidéo : {a.video}  ({w}x{h}, {dur:.1f} s)")
    if a.dry_run:
        print("Dry-run : rien n'a été envoyé.")
        return
    from googleapiclient.discovery import build
    youtube = build("youtube", "v3", credentials=credentials(a.client_secret, a.token))
    resp = upload(youtube, a.video, body)
    vid = resp["id"]
    thumb = a.thumbnail or external_thumb(a.meta)  # priorité : --thumbnail, puis thumbnails/<modèle>/thumbnail.*
    if not thumb and meta.get("thumbnail"):  # miniature fabriquée par tools/make_thumbnail.py
        cand = os.path.join(os.path.dirname(os.path.abspath(a.meta)), meta["thumbnail"])
        thumb = cand if os.path.exists(cand) else None
    if thumb:
        from googleapiclient.http import MediaFileUpload
        try:
            youtube.thumbnails().set(videoId=vid, media_body=MediaFileUpload(thumb)).execute()
            print(f"Miniature ajoutée : {thumb}")
        except Exception as e:  # chaîne non vérifiée, scope insuffisant...
            print(f"Miniature non ajoutée ({e}). Ajoute-la à la main dans YouTube Studio : {thumb}")
    print(f"Envoyée : https://youtube.com/shorts/{vid}  (statut : {resp['status']['privacyStatus']})")


if __name__ == "__main__":
    main()
