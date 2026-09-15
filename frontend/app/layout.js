import "./globals.css";

export const metadata = {
  title: "ChurnCast — Subscriber Churn Prediction",
  description:
    "ChurnCast prototype: subscriber churn prediction on KKBOX-like synthetic data.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
