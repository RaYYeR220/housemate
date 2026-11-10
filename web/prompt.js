// What the approval / question card shows, built from the hub's `prompt` message. No DOM here
// (ui.js draws it), so the tests run this file as it ships.
// An approval card shows Hermes's command exactly as it will run: never cut, never re-spaced.
// Approve is offered only when the hub offers it (Hermes may allow only Deny).

const cap1 = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);

export function promptView(p) {
  const choices = Array.isArray(p.choices) ? p.choices.map(String) : [];
  if (p.kind === 'approval') {
    const canApprove = choices.includes('Approve');
    const details = [];
    if (p.detail) details.push({ text: cap1(String(p.detail)) });
    if (typeof p.command === 'string' && p.command) details.push({ text: p.command, code: true }); // verbatim
    if (!details.length) details.push({ text: String(p.text || '') });
    const buttons = [];
    if (canApprove) buttons.push({ label: 'Approve', value: 'approve', primary: true });
    buttons.push({ label: 'Deny', value: 'deny' });
    return {
      title: p.title || 'Allow this?', details, buttons,
      foot: canApprove ? 'Say "approve" or "deny". No answer means deny.'
        : 'Only Deny is offered for this one. Say "deny", or tap it. No answer means deny.',
    };
  }
  if (p.kind === 'resume') { // Hermes paused this conversation after a restart
    return {
      title: p.title || 'Conversation paused', details: [{ text: String(p.detail || p.text || '') }],
      buttons: [{ label: 'Continue', value: 'continue', primary: true }],
      foot: 'Say "continue", or tap it.',
    };
  }
  const buttons = choices.slice(0, 8).map((c) => ({ label: c, value: c }));
  return { title: String(p.text || 'A question'), details: [], buttons, foot: buttons.length ? 'Say your choice, or tap it.' : 'Say your answer.' };
}
