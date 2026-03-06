## Summary

- Explain what changed
- Explain why it changed
- Mention anything reviewers should focus on

## Testing

- [ ] `python -m unittest discover -s tests -v`
- [ ] `python -m py_compile app.py massscriber/__init__.py massscriber/types.py massscriber/exporters.py massscriber/transcriber.py massscriber/ui.py`
- [ ] Manual verification performed when relevant

## Checklist

- [ ] I updated docs if behavior changed
- [ ] I added or updated tests when needed
- [ ] I verified generated or temporary files are not being committed
