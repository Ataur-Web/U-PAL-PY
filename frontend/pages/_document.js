import { Html, Head, Main, NextScript } from 'next/document';

// custom _document lets us set <html lang>, preconnect to fonts, and drop a
// meta description. Next.js only renders this on the server.
// ref: https://nextjs.org/docs/pages/building-your-application/routing/custom-document

export default function Document() {
  return (
    <Html lang="en" data-theme="light" data-font="editorial">
      <Head>
        {/* preconnect first so the browser opens the TCP+TLS handshake in
            parallel with parsing the rest of <head>. saves ~100ms on cold
            loads. ref: https://web.dev/articles/preconnect-and-dns-prefetch */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500;600&family=JetBrains+Mono:ital,wght@0,400;0,500;1,400&family=Space+Grotesk:wght@400;500;600&family=Space+Mono:ital,wght@0,400;0,700;1,400&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Mono:ital,wght@0,400;0,500;1,400&display=swap"
          rel="stylesheet"
        />
        <meta
          name="description"
          content="U-Pal, a bilingual Welsh/English student assistant for UWTSD. Ask anything about admissions, fees, accommodation, IT support, wellbeing and more."
        />
      </Head>
      <body>
        <Main />
        <NextScript />
      </body>
    </Html>
  );
}
