import { Footer, Layout, Navbar } from 'nextra-theme-docs'
import { Head } from 'nextra/components'
import { getPageMap } from 'nextra/page-map'
import 'nextra-theme-docs/style.css'

export const metadata = {
  title: 'Neuracore Documentation',
  description: 'A powerful robot learning library for data collection, model training, and real-time inference'
}

export default async function RootLayout({ children }) {
  const pageMap = await getPageMap()
  
  return (
    <html lang="en" dir="ltr" suppressHydrationWarning>
      <Head />
      <body>
        <Layout
          navbar={<Navbar logo={<b>Neuracore</b>} projectLink="https://github.com/NeuracoreAI/neuracore" />}
          pageMap={pageMap}
          docsRepositoryBase="https://github.com/NeuracoreAI/neuracore/tree/main/docs"
          footer={<Footer>MIT {new Date().getFullYear()} © Neuracore</Footer>}
        >
          {children}
        </Layout>
      </body>
    </html>
  )
}
