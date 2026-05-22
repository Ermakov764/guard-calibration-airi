# Освобождение диска (безопасно)

**Сейчас:** `/` занят на **92%**, свободно **~8.3 GB** — мало для torch + 7B модели.

## Крупные папки (по `du`, только чтение)

| Путь | Размер | Безопасность очистки |
|------|--------|----------------------|
| `~/.config/Cursor` | **7.5 GB** | Осторожно: кэш; можно Cursor → Clear cache / старые workspace storage |
| `~/snap/telegram-desktop` | **3.9 GB** | `snap remove` только если не пользуетесь |
| `~/.cache/google-chrome` | **~1 GB** | Да: настройки Chrome → очистить кэш |
| `~/.config/Code/CachedExtensionVSIXs` | **1.1 GB** | Да: старые VSIX кэши |
| `~/.gradle` | **893 MB** | Да: `rm -rf ~/.gradle/caches` (перекачает при сборке) |
| `~/.npm` | **464 MB** | Да: `npm cache clean --force` |
| `~/.m2` | **264 MB** | Да: старые артефакты Maven |
| `~/Downloads` | **2.5 GB** | Вручную удалить ненужное |

## Команды (выполняйте сами — ничего не удалено автоматически)

```bash
# Посмотреть, что займёт больше всего
du -h --max-depth=1 ~ | sort -hr | head -20

# Кэш pip
pip cache purge

# Gradle (если не нужен срочно)
rm -rf ~/.gradle/caches/

# npm
npm cache clean --force

# Старый кэш Chrome (закройте браузер)
rm -rf ~/.cache/google-chrome/Default/Cache/*
```

**Цель перед ML:** **≥20 GB** свободно на `/`.

## НЕ удалять без понимания

- `~/Desktop/Статья AIRI` — ваш проект
- `~/.config/Cursor/User` целиком — настройки IDE
- `/boot`, системные каталоги
