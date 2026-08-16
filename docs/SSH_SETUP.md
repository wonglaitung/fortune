# SSH 认证配置指南

> 适用于在新电脑上配置 GitHub SSH 认证，替代 HTTPS 令牌方式。
> **每台电脑生成自己的新密钥**，不要复制私钥文件（更安全）。

---

## 为什么用 SSH 代替 HTTPS 令牌

- HTTPS 地址若内嵌令牌（`https://ghp_xxx@github.com/...`），令牌一旦在会话/日志中明文出现即泄露。
- SSH 使用密钥对认证，私钥仅存本机，公钥可在 GitHub 添加多个。
- 本项目远程地址已切换为 `git@github.com:wonglaitung/fortune.git`。

---

## 一、生成密钥（在目标电脑上）

```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
ssh-keygen -t ed25519 -C "marcowong@<这台机器的名字>" -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub   # 复制输出
```

> `-N ""` 表示无密码短语；如想更安全可去掉并输入口令。
> 注释 `-C` 建议标明机器名，方便在 GitHub 上区分多个密钥。

---

## 二、添加公钥到 GitHub

打开 **https://github.com/settings/ssh/new**

1. Title 填任意名字（如 `my-laptop`）
2. Key type 选 **Authentication Key**
3. 粘贴公钥 → **Add SSH key**

> 同一账户可添加多个公钥，每台电脑一个。

---

## 三、写 SSH 配置（可选，但推荐）

```bash
cat > ~/.ssh/config << 'EOF'
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    AddKeysToAgent yes
EOF
chmod 600 ~/.ssh/config
```

> `AddKeysToAgent yes` 让 SSH 连接时自动把密钥加入 agent；
> `IdentitiesOnly yes` 确保只用指定密钥，避免 agent 里多个密钥时匹配错。

---

## 四、启动 agent 并测试

```bash
eval "$(ssh-agent -s)" && ssh-add ~/.ssh/id_ed25519
ssh -T git@github.com
# 看到 "Hi wonglaitung! You've successfully authenticated" 即成功
# （退出码 1 是 GitHub 对无 shell 访问的正常响应，不是错误）
```

---

## 五、切换仓库远程地址

```bash
git remote set-url origin git@github.com:wonglaitung/fortune.git
git fetch && git status   # 验证
```

---

## 六、清理旧令牌（原电脑若曾用 HTTPS 令牌）

```bash
# 1. 远程地址去掉令牌
git remote set-url origin https://github.com/wonglaitung/fortune.git

# 2. 清除明文凭证存储中的令牌
grep -v "<token>" ~/.git-credentials > /tmp/cred.tmp && mv /tmp/cred.tmp ~/.git-credentials

# 3. 吊销令牌（必须手动）：https://github.com/settings/tokens → Delete
```

> ⚠️ 令牌一旦明文暴露（会话输出、日志、git config）应立即吊销。

---

## 关键安全要点

| 事项 | 说明 |
|------|------|
| 每台电脑一个密钥 | 各自 `ssh-keygen`，各自把 `.pub` 加到 GitHub |
| 私钥不外传 | `id_ed25519` 永不复制、永不上传；只传 `.pub` |
| 令牌吊销 | 泄露过的令牌在 https://github.com/settings/tokens 删除 |
| 权限最小化 | 新建 PAT 时仅授予 `repo` 等最小 scope |
| 凭证存储 | 避免使用 `git config credential.helper store`（明文） |
