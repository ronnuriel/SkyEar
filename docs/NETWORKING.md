# SkyEar Networking Runbook

This guide explains how to connect SkyEar stations to a central server when they are not all running on the same computer.

SkyEar sends compact JSON events and heartbeats from each station to the server. Raw audio is not sent to the server.

## Ports And Endpoints

Default server:

```bash
skyear-server --host 0.0.0.0 --port 8080
```

Main station endpoints:

- `GET /health`
- `GET /stations/health`
- `POST /events`
- `POST /stations/heartbeat`
- `GET /fusion`
- `GET /dashboard/state`
- `GET /dashboard/live`

The operator dashboard uses `/dashboard/state` in normal mode and the smaller `/dashboard/live` endpoint in live mode to reduce request volume during simulations.

Station config usually points at:

```yaml
server:
  url: http://SERVER_IP:8080/events
```

Before starting a remote station, test connectivity:

```bash
skyear-check-server --url http://SERVER_IP:8080
```

With auth:

```bash
skyear-check-server \
  --url http://SERVER_IP:8080 \
  --api-token "$SKYEAR_API_TOKEN" \
  --hmac-secret "$SKYEAR_HMAC_SECRET"
```

## Same-LAN Deployment

Use this when the server and station are on the same Wi-Fi/Ethernet network.

On the server computer:

```bash
skyear-server --host 0.0.0.0 --port 8080
```

Find the server IP:

```bash
ipconfig getifaddr en0
```

or on Linux:

```bash
hostname -I
```

On the station computer:

```bash
skyear-check-server --url http://SERVER_IP:8080
skyear-station --config configs/config_station_remote.yaml
```

Firewall checklist:

- Allow inbound TCP `8080` on the server computer.
- Keep all machines on the same LAN/VLAN.
- Do not use `127.0.0.1` from the station unless the server is running on that same station computer.

## Tailscale Or WireGuard Deployment

Use this for field stations over different networks. This is the preferred remote deployment style because it avoids exposing SkyEar directly to the public internet.

Tailscale:

1. Install Tailscale on the server and station computers.
2. Join both to the same tailnet.
3. Use the server Tailscale IP or MagicDNS name:

```yaml
server:
  url: http://SERVER_TAILSCALE_IP:8080/events
```

Then test:

```bash
skyear-check-server --url http://SERVER_TAILSCALE_IP:8080
```

WireGuard:

1. Create a private WireGuard network for server and stations.
2. Allow TCP `8080` through the WireGuard interface only.
3. Use the server WireGuard IP in station configs.

Security note: even inside a VPN, use SkyEar token or HMAC auth if multiple users/devices can reach the network.

## Temporary Tunnel With Ngrok Or Cloudflared

Use this only for short demos or debugging. Tunnels expose the server through a public URL.

Ngrok:

```bash
skyear-server --host 127.0.0.1 --port 8080
ngrok http 8080
```

Station config:

```yaml
server:
  url: https://YOUR-NGROK-DOMAIN.ngrok-free.app/events
  api_token: change-me
  hmac_secret: change-me-too
```

Cloudflared:

```bash
skyear-server --host 127.0.0.1 --port 8080
cloudflared tunnel --url http://127.0.0.1:8080
```

Station config:

```yaml
server:
  url: https://YOUR-CLOUDFLARED-DOMAIN.trycloudflare.com/events
  api_token: change-me
  hmac_secret: change-me-too
```

For any public tunnel, auth is mandatory.

## Nginx Reverse Proxy Deployment

Use this when you want a stable domain, TLS, and a production-style endpoint.

Run SkyEar API locally on the server:

```bash
skyear-server --host 127.0.0.1 --port 8080
```

Example Nginx site:

```nginx
server {
    listen 443 ssl;
    server_name skyear.example.com;

    ssl_certificate /etc/letsencrypt/live/skyear.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/skyear.example.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Station config:

```yaml
server:
  url: https://skyear.example.com/events
  api_token: change-me
  hmac_secret: change-me-too
```

Test:

```bash
skyear-check-server \
  --url https://skyear.example.com \
  --api-token change-me \
  --hmac-secret change-me-too
```

## Auth Requirements Outside Localhost

If the server is reachable by anything other than your own local machine, configure auth.

On the server:

```bash
export SKYEAR_API_TOKEN="change-me"
export SKYEAR_HMAC_SECRET="change-me-too"
skyear-server --host 0.0.0.0 --port 8080
```

On every station:

```yaml
server:
  url: http://SERVER_IP:8080/events
  api_token: change-me
  hmac_secret: change-me-too
```

Auth behavior:

- If neither env var is set on the server, local development is unauthenticated.
- If `SKYEAR_API_TOKEN` is set, stations must send the token.
- If `SKYEAR_HMAC_SECRET` is set, stations must sign request bodies.
- If both are set, both must pass.

## Startup Behavior

Stations run a startup server connectivity check by default:

```yaml
server:
  startup_check_enabled: true
  startup_check_timeout_sec: 2.0
```

If the server is unreachable, the station prints a warning and continues in local monitor mode. Posting and heartbeat attempts keep retrying during normal operation.

## Troubleshooting

`skyear-check-server` cannot reach `/health`:

- Check server is running.
- Check host/IP and port.
- Check firewall.
- From another machine, do not use `127.0.0.1`.

`/health` works but heartbeat fails:

- Check token/HMAC config.
- Confirm station and server use the same secrets.
- If using HMAC behind a proxy, make sure the proxy does not rewrite the JSON body.

Station local monitor works but dashboard is empty:

- The station is running locally, but central POSTs are failing.
- Check station console for `[WARN] send failed`.
- Run `skyear-check-server --url http://SERVER:8080` from the station machine.
