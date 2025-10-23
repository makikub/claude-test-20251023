---
name: hello
description: read prompt.md, and prints "hello world" to stdout using Python
---

This skill prints "hello world" to stdout using Python.

## スキルの構成

このスキルは以下のファイルで構成されています：

- **SKILL.md** (このファイル): スキルのメタデータと説明
- **prompt.md** (必須): Claude Codeが実行する指示が記載されたプロンプトファイル
- **hello.py**: 実際に"hello world"を出力するPythonスクリプト

## 実行フロー

このスキルは以下の流れで実行されます：

1. **スキル呼び出し**: ユーザーが `Skill` ツールで "hello" を呼び出す
2. **prompt.mdの自動読み込み**: Claude Codeシステムが自動的に `prompt.md` を読み込み、その内容を展開する
3. **指示の実行**: Claude Codeは `prompt.md` に記載された指示に従って動作する
4. **hello.pyの実行**: `prompt.md` の指示に基づいて、Bashツールで以下のコマンドを実行
   ```bash
   python3 .claude/skills/hello/hello.py
   ```
5. **結果の出力**: スクリプトが標準出力に "hello world" を出力し、タイムスタンプ付きの `hello.txt` ファイルを `.claude/` ディレクトリに作成
6. **結果の報告**: 実行結果がユーザーに報告される

## 重要事項

- **prompt.mdは必須**: このファイルが存在しない場合、スキルは正常に動作しません
- **自動読み込み**: prompt.mdはClaude Codeシステムによって自動的に読み込まれるため、手動での読み込み操作は不要です
- **トークン消費**: prompt.mdには大量のドキュメントが含まれており、約60,000トークンを消費します（トークン消費のデモンストレーション用）
