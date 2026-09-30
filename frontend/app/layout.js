import "./globals.css";
import Shell from "./components/Shell";

export const metadata = {
  title: "ChurnCast \u2014 Subscriber Churn Intelligence",
  description:
    "ChurnCast prototype: subscriber churn prediction on KKBOX-like synthetic data.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <Shell>{children}</Shell>
      </body>
    </html>
  );
}
