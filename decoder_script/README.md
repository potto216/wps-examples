# How to Use This Extension
1. Create a new folder for your extension (e.g. decoder-script-syntax).
2. In that folder, create the files as shown above:
    * package.json
    * language-configuration.json
    * Create a subfolder named syntaxes and inside it add the file decoder.tmLanguage.json
3. Open this folder in Visual Studio Code.
4. Press F5 to launch a new Extension Development Host.
5. Open or create a file with one of the registered extensions (e.g. example.dec), and you should see the keywords highlighted according to the grammar rules.

You can now adjust the grammar (the list of keywords, regexes, etc.) to suit your full DecoderScript syntax requirements.

This sample extension is a starting point and can be further enhanced with additional scopes, language features, and improved patterns as needed.

I just want to setup a The files we shared set up a TextMate grammar and language configuration that provide color highlighting and file associations for your DecoderScript files.

https://code.visualstudio.com/api/language-extensions/syntax-highlight-guide