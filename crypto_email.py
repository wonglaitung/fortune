import os
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def get_cryptocurrency_prices():
    # 注意：原 URL 末尾有空格，已修正
    url = "https://api.coingecko.com/api/v3/simple/price"
    
    params = {
        'ids': 'bitcoin,ethereum',
        'vs_currencies': 'usd,hkd',
        'include_24hr_change': 'true'
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching cryptocurrency prices: HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"Exception during API request: {e}")
        return None

def send_email(to, subject, text, html):
    smtp_server = "smtp.mail.yahoo.com"
    smtp_port = 587
    smtp_user = os.environ.get("YAHOO_EMAIL")
    smtp_pass = os.environ.get("YAHOO_APP_PASSWORD")
    sender_email = smtp_user

    if not smtp_user or not smtp_pass:
        print("Error: Missing YAHOO_EMAIL or YAHOO_APP_PASSWORD in environment variables.")
        return False

    msg = MIMEMultipart("alternative")
    msg['From'] = f'"wonglaitung" <{sender_email}>'
    msg['To'] = to
    msg['Subject'] = subject

    msg.attach(MIMEText(text, "plain"))
    msg.attach(MIMEText(html, "html"))

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(sender_email, to, msg.as_string())
        server.quit()
        print("✅ Email sent successfully!")
        return True
    except Exception as e:
        print(f"❌ Error sending email: {e}")
        return False

# === 主逻辑 ===
if __name__ == "__main__":
    prices = get_cryptocurrency_prices()

    if prices is None:
        print("Failed to fetch prices. Exiting.")
        exit(1)

    subject = "Bitcoin and Ethereum Price Update"

    text = ""
    html = "<html><body>"

    # Bitcoin
    if 'bitcoin' in prices:
        btc = prices['bitcoin']
        btc_usd = btc['usd']
        btc_hkd = btc['hkd']
        btc_change = btc.get('usd_24h_change', 0.0)
        text += f"Bitcoin price: ${btc_usd:,.2f} USD ({btc_change:.2f}% 24h)\n"
        text += f"Bitcoin price: ${btc_hkd:,.2f} HKD\n\n"
        html += f"<p><strong>Bitcoin</strong><br>"
        html += f"Price: ${btc_usd:,.2f} USD ({btc_change:.2f}% 24h)<br>"
        html += f"Price: ${btc_hkd:,.2f} HKD</p>"

    # Ethereum
    if 'ethereum' in prices:
        eth = prices['ethereum']
        eth_usd = eth['usd']
        eth_hkd = eth['hkd']
        eth_change = eth.get('usd_24h_change', 0.0)
        text += f"Ethereum price: ${eth_usd:,.2f} USD ({eth_change:.2f}% 24h)\n"
        text += f"Ethereum price: ${eth_hkd:,.2f} HKD\n"
        html += f"<p><strong>Ethereum</strong><br>"
        html += f"Price: ${eth_usd:,.2f} USD ({eth_change:.2f}% 24h)<br>"
        html += f"Price: ${eth_hkd:,.2f} HKD</p>"

    html += "</body></html>"

    # 获取收件人（默认 fallback）
    recipient = os.environ.get("RECIPIENT_EMAIL", "wonglaitung@google.com")

    print("📧 Sending email to:", recipient)
    print("📝 Subject:", subject)
    print("📄 Text preview:\n", text)

    success = send_email(recipient, subject, text, html)
    if not success:
        exit(1)
