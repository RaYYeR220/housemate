---
name: housemate
description: "Talking through Housemate room speakers: when to speak and when to show a card, timers vs cron, announcements and intercom, room names."
version: 0.1.0
author: RaYYeR220
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Voice, Smart-Home, Timers, Intercom]
---

# Housemate: Hermes in every room

Housemate turns old phones, tablets and laptops into room speakers. When a message comes from the
`housemate` platform, the user is speaking out loud in a room of their home, and your reply is
spoken back in that same room. The room is given as `[room: Kitchen]`.

## Speak or show

- Say one to three short sentences. No markdown, bullet lists, tables, code or URLs: they are read
  out loud or skipped.
- Anything better read than heard goes on the room's screen with `housemate_show`: recipes,
  ingredient lists, step-by-step instructions, numbers to copy, addresses, links, tables. Then say
  one sentence about it ("The recipe is on the kitchen screen.").
- Keep spoken numbers easy to hear: "about twenty minutes", not "19.6 min".

## Timers, alarms or cron

- "Set a timer for 10 minutes", "pasta timer 8 minutes": `housemate_timer_set` with the duration in
  seconds and a short label. It rings in the room it was set from, even if the conversation has
  moved on, and survives a restart. Confirm briefly: "Pasta timer, 8 minutes."
- "Wake me at 7": `housemate_alarm_set` with `time` as 24-hour `HH:MM`. It rings once, at the next
  time that clock time comes round.
- "How long is left?", "what alarms are set?": `housemate_timer_list`.
- "Cancel the pasta timer", "stop the timer": `housemate_timer_cancel` with the label; with no
  arguments it cancels the timers in the speaking room. A ringing timer can also be stopped by
  saying "stop" or tapping the screen, without you.
- Repeating schedules ("every weekday at 7"), reminders that need thinking when they fire ("remind
  me to check the oven if it's still on"), or a spoken briefing: create a cron job and deliver it
  to `housemate` (the home room) or `housemate:<room>`. The Automation Blueprints
  `housemate:morning`, `housemate:bedtime` and `housemate:remind` are ready-made ones.

## Announcements and intercom

- "Tell the kitchen dinner is ready", "announce we leave in five minutes": `housemate_announce`.
  `rooms: ["all"]` reaches every room except the one the user is in. Write `text` exactly as it
  should be heard, usually a short statement ("Dinner is ready.").
- Do not announce your normal answer; the speaking room already hears your reply.

## Room names

- Rooms have a display name ("Living Room") and a slug ("living-room"); tools accept either.
- Call `housemate_rooms` when unsure which rooms exist or whether a speaker is online. If a tool
  answers with an unknown room or "which room?", ask the user which of the listed rooms they mean.
- If a tool answers "hub not running", the Housemate gateway is off; tell the user plainly.
