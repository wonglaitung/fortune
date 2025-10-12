import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import os
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def get_hk_ipo_info_from_aastocks():
    """
    通过爬取AAStocks网站获取香港股市IPO信息
    """
    url = "http://www.aastocks.com/tc/stocks/market/ipo/upcomingipo/company-summary"
    
    # 设置请求头，模拟浏览器访问
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        # 发送HTTP请求
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = 'utf-8'  # 设置编码
        
        if response.status_code == 200:
            # 使用BeautifulSoup解析HTML内容
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 查找同期新股部分的公司信息
            upcoming_ipo_data = []
            
            # 查找包含"同期新股"的表格
            tables = soup.find_all('table')
            for table in tables:
                # 检查表格是否包含"同期新股"和"公司名稱"
                if table.find(string=lambda text: text and '同期新股' in text) and table.find(string=lambda text: text and '公司名稱' in text):
                    # 遍历表格行
                    rows = table.find_all('tr')
                    for row in rows:
                        # 查找包含公司信息的行
                        cells = row.find_all(['td', 'th'])
                        cell_texts = [cell.get_text(strip=True) for cell in cells]
                        
                        # 检查是否包含日期格式来识别公司信息行
                        if any(re.match(r'\d{4}/\d{2}/\d{2}', text) for text in cell_texts):
                            # 提取公司名称、上市日期和行业
                            # 假设格式为: 公司名称, 上市日期, 行业
                            if len(cell_texts) >= 3:
                                company_name = cell_texts[0]
                                listing_date = cell_texts[1]
                                industry = cell_texts[2]
                                
                                # 只有当公司名称、日期、行业都存在时才添加
                                if company_name and listing_date and industry:
                                    # 过滤掉包含多余文本的行
                                    if not any(keyword in company_name for keyword in ['招股日程', '同期新股公司名稱', '招股日期', '公司名稱']):
                                        company_info = {
                                            '公司名称': company_name,
                                            '上市日期': listing_date,
                                            '行业': industry
                                        }
                                        upcoming_ipo_data.append(company_info)
            
            # 查找IPO基本信息
            ipo_basic_info = {}
            
            # 查找包含IPO基本信息的表格
            for table in tables:
                # 遍历表格行
                rows = table.find_all('tr')
                for row in rows:
                    # 查找包含信息的行
                    cells = row.find_all(['td', 'th'])
                    cell_texts = [cell.get_text(strip=True) for cell in cells]
                    
                    # 提取关键信息
                    if len(cell_texts) >= 2:
                        key = cell_texts[0]
                        value = cell_texts[1]
                        
                        if '招股日期' in key:
                            ipo_basic_info['招股日期'] = value
                        elif '每手股數' in key:
                            ipo_basic_info['每手股数'] = value
                        elif '招股價' in key:
                            ipo_basic_info['招股价格'] = value
                        elif '入場費' in key:
                            ipo_basic_info['入场费'] = value
                        elif '退票寄發日期' in key:
                            ipo_basic_info['暗盘日期'] = value
            
            # 创建存储IPO信息的列表
            result_data = []
            
            # 添加IPO基本信息
            if ipo_basic_info:
                result_data.append({
                    '类别': 'IPO基本信息',
                    '公司名称': '',
                    '招股日期': ipo_basic_info.get('招股日期', ''),
                    '每手股数': ipo_basic_info.get('每手股数', ''),
                    '招股价格': ipo_basic_info.get('招股价格', ''),
                    '入场费': ipo_basic_info.get('入场费', ''),
                    '暗盘日期': ipo_basic_info.get('暗盘日期', '')
                })
            
            # 添加同期新股部分的公司
            for info in upcoming_ipo_data:
                result_data.append({
                    '类别': '即将上市',
                    '公司名称': info['公司名称'],
                    '上市日期': info['上市日期'],
                    '行业': info['行业']
                })
            
            # 如果找到了相关信息，返回DataFrame
            if result_data:
                return pd.DataFrame(result_data)
            else:
                print("未能从页面中提取到有效的IPO信息")
                return None
        else:
            print(f"获取网页失败，状态码：{response.status_code}")
            return None
            
    except Exception as e:
        print(f"通过AAStocks获取IPO信息时发生错误: {e}")
        return None

def print_ipo_data(data):
    """
    打印IPO数据
    """
    if data is not None and not data.empty:
        print("\n=== 最新港股IPO信息 ===")
        print(data.to_string(index=False))
    else:
        print("未能获取到IPO数据")

def save_ipo_data_to_csv(data, filename=None):
    """
    将IPO数据保存为CSV文件
    """
    if filename is None:
        filename = f"hk_ipo_aastocks_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    if data is not None and not data.empty:
        # 保存到CSV文件
        data.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"IPO数据已保存到 {filename}")
        return filename
    else:
        print("没有数据可保存")
        return None

def send_ipo_email(data, to):
    """
    发送IPO信息邮件
    """
    smtp_server = "smtp.mail.yahoo.com"
    smtp_port = 587
    smtp_user = os.environ.get("YAHOO_EMAIL")
    smtp_pass = os.environ.get("YAHOO_APP_PASSWORD")
    sender_email = smtp_user

    if not smtp_user or not smtp_pass:
        print("Error: Missing YAHOO_EMAIL or YAHOO_APP_PASSWORD in environment variables.")
        return False

    # 如果to是字符串，转换为列表
    if isinstance(to, str):
        to = [to]

    subject = "最新港股IPO信息"
    
    # 生成文本和HTML内容
    text = "最新港股IPO信息:\n\n"
    html = "<html><body><h2>最新港股IPO信息</h2>"
    
    if data is not None and not data.empty:
        # 添加文本内容
        text += data.to_string(index=False) + "\n\n"
        
        # 添加HTML内容
        html += data.to_html(index=False, escape=False)
    else:
        text += "未能获取到IPO数据\n\n"
        html += "<p>未能获取到IPO数据</p>"
    
    html += "</body></html>"

    msg = MIMEMultipart("alternative")
    msg['From'] = f'"wonglaitung" <{sender_email}>'
    msg['To'] = ", ".join(to)
    msg['Subject'] = subject

    msg.attach(MIMEText(text, "plain"))
    msg.attach(MIMEText(html, "html"))

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(sender_email, to, msg.as_string())
        server.quit()
        print("✅ IPO信息邮件发送成功!")
        return True
    except Exception as e:
        print(f"❌ 发送邮件时出错: {e}")
        return False

# === 主逻辑 ===
if __name__ == "__main__":
    print("正在通过AAStocks网站获取最新的港股IPO信息...")
    
    # 获取IPO数据
    ipo_data = get_hk_ipo_info_from_aastocks()
    
    if ipo_data is not None:
        # 打印IPO数据
        print_ipo_data(ipo_data)
        
        # 保存到CSV文件
        save_ipo_data_to_csv(ipo_data)
        
        # 发送邮件
        recipient_env = os.environ.get("RECIPIENT_EMAIL", "wonglaitung@google.com")
        
        # 如果环境变量中有多个收件人（用逗号分隔），则拆分为列表
        if ',' in recipient_env:
            recipients = [recipient.strip() for recipient in recipient_env.split(',')]
        else:
            recipients = [recipient_env]
        
        print("📧 发送邮件到:", ", ".join(recipients))
        send_ipo_email(ipo_data, recipients)
    else:
        print("获取IPO数据失败")