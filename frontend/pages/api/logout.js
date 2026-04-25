'use strict';
// GET /api/logout. there is no server-side session for Basic Auth, so the
// only way to "log out" is to force the browser to drop its cached
// credentials by sending a fresh 401 with a WWW-Authenticate header. the
// user's next admin request will then be re-prompted.
// ref: https://developer.mozilla.org/en-US/docs/Web/HTTP/Authentication#basic_authentication_scheme

export default function handler(req, res) {
  res.setHeader('WWW-Authenticate', 'Basic realm="U-Pal Admin"');
  return res.status(401).json({ message: 'Logged out' });
}
