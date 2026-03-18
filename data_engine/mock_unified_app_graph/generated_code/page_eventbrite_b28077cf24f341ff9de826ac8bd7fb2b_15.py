# page_id: page_eventbrite_b28077cf24f341ff9de826ac8bd7fb2b_15
# screenshot: 2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-17.png
# step_index: 15/16
# task: Open Eventbrite. Explore 'Wellness' events in Washington. Filter to only show free events. Add the first non-promoted event to favorite and follow the organizer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural UI elements for Eventbrite-like page
w, h = canvas.size

# 1) Overall page background (very light off-white)
draw.rectangle([(0, 0), (w, h)], fill="#FBFBFC")

# 2) Status bar area (top ~60px) - muted dark gray/green
status_h = 60
draw.rectangle([(0, 0), (w, status_h)], fill="#7E867F")

# 3) Notification banner under status bar (mint green)
notif_y0 = status_h
notif_y1 = status_h + 120
draw.rectangle([(0, notif_y0), (w, notif_y1)], fill="#E7F6EE")
# subtle bottom divider for banner
draw.line([(24, notif_y1), (w-24, notif_y1)], fill="#D7E9DE", width=1)

# 4) Hero image area with a subtle dark-to-light vertical gradient (image sits on top)
hero_y0 = notif_y1
hero_y1 = hero_y0 + 260
top_color = (48, 48, 48)   # dark
bot_color = (230, 230, 230)  # light gray
# draw gradient bands
for i in range(hero_y1 - hero_y0):
    t = i / max(1, (hero_y1 - hero_y0 - 1))
    r = int(top_color[0] * (1-t) + bot_color[0] * t)
    g = int(top_color[1] * (1-t) + bot_color[1] * t)
    b = int(top_color[2] * (1-t) + bot_color[2] * t)
    draw.line([(0, hero_y0 + i), (w, hero_y0 + i)], fill=(r, g, b))

# subtle dark overlay band at bottom of hero for contrast (where scrub bar sits)
overlay_h = 36
draw.rectangle([(0, hero_y1 - overlay_h), (w, hero_y1)], fill=(40, 40, 40))

# horizontal small scrub bars placeholders (background bars only, no icons/text)
scrub_y = hero_y1 - int(overlay_h / 2) - 6
bar_w = int(w * 0.12)
gap = int(w * 0.03)
x_start = int((w - (bar_w*4 + gap*3)) / 2)
for i in range(4):
    x0 = x_start + i*(bar_w + gap)
    x1 = x0 + bar_w
    draw.rounded_rectangle([(x0, scrub_y - 6), (x1, scrub_y + 6)], radius=6, fill="#BFBFBF")

# 5) Main content area (starts just below hero) - keep white base (already set),
# add subtle top divider shadow
content_y0 = hero_y1
draw.line([(24, content_y0), (w-24, content_y0)], fill="#E6E6E6", width=2)

# 6) Organizer card / profile background (rounded card behind avatar and Follow button)
card_x0 = 48
card_x1 = w - 48
card_y0 = content_y0 + 260
card_y1 = card_y0 + 120
# light rounded card background
draw.rounded_rectangle([(card_x0, card_y0), (card_x1, card_y1)], radius=20, fill="#F6F5F8")
# subtle inner shadow/top highlight
draw.line([(card_x0+8, card_y0+8), (card_x1-8, card_y0+8)], fill="#EFEFF1", width=1)

# 7) Small divider under event info / refund policy area
sep_y = card_y1 + 180
draw.line([(48, sep_y), (w-48, sep_y)], fill="#ECECEC", width=1)

# 8) "About this event" pill/tag background (rounded pill)
pill_x0 = 48
pill_y0 = sep_y + 60
pill_w = 360
pill_h = 56
draw.rounded_rectangle([(pill_x0, pill_y0), (pill_x0 + pill_w, pill_y0 + pill_h)], radius=28, fill="#EEF2F7")

# 9) Additional faint section divider under description
desc_sep_y = pill_y0 + 140
draw.line([(48, desc_sep_y), (w-48, desc_sep_y)], fill="#F0F0F2", width=1)

# 10) Ticket selection box (rounded white box with colored stroke) - keep content area clear for pasted controls
ticket_x0 = 48
ticket_x1 = w - 48
ticket_y1 = desc_sep_y + 450
ticket_h = 220
ticket_y0 = ticket_y1 - ticket_h
# outer border (rounded)
border_color = "#3750F0"
draw.rounded_rectangle([(ticket_x0, ticket_y0), (ticket_x1, ticket_y1)], radius=18, fill="#FFFFFF", outline=border_color, width=8)
# inner subtle background strip near top of ticket box (for title area background separation)
inner_strip_h = 64
draw.rectangle([(ticket_x0+8, ticket_y0+8), (ticket_x1-8, ticket_y0+8+inner_strip_h)], fill="#FFFFFF")

# small shadow under ticket box
shadow_y0 = ticket_y1
for i, a in enumerate(range(6)):
    alpha = 220 - i*36
    shade = int(230 - i*8)
    draw.line([(ticket_x0+4, shadow_y0 + i), (ticket_x1-4, shadow_y0 + i)], fill=(shade, shade, shade), width=1)

# 11) Subtle separator above the reserve button area (so the orange CTA sits distinct)
cta_sep_y = ticket_y1 + 120
draw.line([(48, cta_sep_y), (w-48, cta_sep_y)], fill="#EEE0D8", width=1)

# 12) Bottom safe area background (where CTA sits) - keep neutral but avoid painting over the CTA region itself.
# We draw a large soft background strip below the CTA separator for visual separation, but stop short of the CTA bounding box.
bottom_strip_y0 = cta_sep_y
bottom_strip_y1 = h - 140  # leave space; the real CTA will be pasted near the bottom
draw.rectangle([(0, bottom_strip_y0), (w, bottom_strip_y1)], fill="#FFFFFF")

# 13) Fine vertical margins guides (very subtle) - purely decorative separators
draw.line([(48, hero_y1 + 8), (48, h-200)], fill="#FAFAFB", width=1)
draw.line([(w-48, hero_y1 + 8), (w-48, h-200)], fill="#FAFAFB", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_15_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-17/00_icon_Follow.png
try:
    _c0 = get_crop(0, 331, 144)
    canvas.paste(_c0, (1013, 1068), _c0)
except Exception:
    pass
layout["Follow"] = [1013, 1068, 1344, 1212]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_15_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-17/01_icon_Decrease.png
try:
    _c1 = get_crop(1, 99, 96)
    canvas.paste(_c1, (996, 2444), _c1)
except Exception:
    pass
layout["Decrease"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_15_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-17/02_icon_Reserve_a_spot.png
try:
    _c2 = get_crop(2, 1296, 132)
    canvas.paste(_c2, (72, 2756), _c2)
except Exception:
    pass
layout["Reserve_a_spot"] = [72, 2756, 1368, 2888]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_15_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-17/03_icon_Increase.png
try:
    _c3 = get_crop(3, 96, 96)
    canvas.paste(_c3, (1224, 2444), _c3)
except Exception:
    pass
layout["Increase"] = [1224, 2444, 1320, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_15_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-17/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 49, 67)
    canvas.paste(_c4, (1154, 1), _c4)
except Exception:
    pass
layout["icon_4"] = [1154, 1, 1203, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_15_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-17/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 92, 102)
    canvas.paste(_c5, (1108, 2442), _c5)
except Exception:
    pass
layout["icon_5"] = [1108, 2442, 1200, 2544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_15_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-17/06_icon_Health_Wellness.png
try:
    _c6 = get_crop(6, 234, 119)
    canvas.paste(_c6, (48, 2205), _c6)
except Exception:
    pass
layout["Health_&_Wellness"] = [48, 2205, 282, 2324]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_15_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-17/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 58, 58)
    canvas.paste(_c7, (311, 4), _c7)
except Exception:
    pass
layout["icon_7"] = [311, 4, 369, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_15_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-17/08_icon_Dismiss_notification.png
try:
    _c8 = get_crop(8, 142, 142)
    canvas.paste(_c8, (1251, 97), _c8)
except Exception:
    pass
layout["Dismiss_notification"] = [1251, 97, 1393, 239]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_15_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-17/09_icon_4.45.png
try:
    _c9 = get_crop(9, 63, 63)
    canvas.paste(_c9, (179, 1), _c9)
except Exception:
    pass
layout["4.45"] = [179, 1, 242, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_15_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-17/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 64, 64)
    canvas.paste(_c10, (1212, 1), _c10)
except Exception:
    pass
layout["icon_10"] = [1212, 1, 1276, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_15_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-17/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 46, 60)
    canvas.paste(_c11, (1325, 4), _c11)
except Exception:
    pass
layout["icon_11"] = [1325, 4, 1371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_15_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-17/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 47, 53)
    canvas.paste(_c12, (251, 7), _c12)
except Exception:
    pass
layout["icon_12"] = [251, 7, 298, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_15_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-17/13_icon_4.45.png
try:
    _c13 = get_crop(13, 62, 65)
    canvas.paste(_c13, (113, 0), _c13)
except Exception:
    pass
layout["4.45"] = [113, 0, 175, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_15_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-17/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 62, 63)
    canvas.paste(_c14, (1251, 1), _c14)
except Exception:
    pass
layout["icon_14"] = [1251, 1, 1313, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_15_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-17/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 52, 62)
    canvas.paste(_c15, (382, 2), _c15)
except Exception:
    pass
layout["icon_15"] = [382, 2, 434, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_15_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-17/16_icon_Personal_health.png
try:
    _c16 = get_crop(16, 234, 119)
    canvas.paste(_c16, (48, 2205), _c16)
except Exception:
    pass
layout["Personal_health"] = [48, 2205, 282, 2324]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_15_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-17/17_icon_The_organizer_will_review_refund_request.png
try:
    _c17 = get_crop(17, 1344, 144)
    canvas.paste(_c17, (48, 1295), _c17)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1295, 1392, 1439]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_15_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-17/18_icon_4.45.png
try:
    _c18 = get_crop(18, 89, 59)
    canvas.paste(_c18, (17, 5), _c18)
except Exception:
    pass
layout["4.45"] = [17, 5, 106, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_15_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-17/19_icon_4.45.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (36, 108), _c19)
except Exception:
    pass
layout["4.45"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_15_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-17/20_icon_Danielle_Smith.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (96, 1067), _c20)
except Exception:
    pass
layout["Danielle_Smith"] = [96, 1067, 240, 1211]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_15_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-17/21_icon_Free.png
try:
    _c21 = get_crop(21, 75, 72)
    canvas.paste(_c21, (249, 2588), _c21)
except Exception:
    pass
layout["Free"] = [249, 2588, 324, 2660]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_15_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-17/22_icon_Free.png
try:
    _c22 = get_crop(22, 135, 107)
    canvas.paste(_c22, (99, 2574), _c22)
except Exception:
    pass
layout["Free"] = [99, 2574, 234, 2681]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_15_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-17/23_text_We_ve_added_the_event_to_your_shortlist.png
try:
    _c23 = get_crop(23, 144, 144)
    canvas.paste(_c23, (36, 108), _c23)
except Exception:
    pass
layout["We've_added_the_event_to_"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_15_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-17/24_text_Friday_May_10_._H_00_AM.png
try:
    _c24 = get_crop(24, 314, 144)
    canvas.paste(_c24, (288, 1068), _c24)
except Exception:
    pass
layout["Friday;_May_10_._H:00_AM"] = [288, 1068, 602, 1212]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_15_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-17/25_text_Wellness_in_Action_Core_Work.png
try:
    _c25 = get_crop(25, 314, 144)
    canvas.paste(_c25, (288, 1068), _c25)
except Exception:
    pass
layout["Wellness_in_Action:_Core_"] = [288, 1068, 602, 1212]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_15_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-17/26_text_Danielle_Smith.png
try:
    _c26 = get_crop(26, 314, 144)
    canvas.paste(_c26, (288, 1068), _c26)
except Exception:
    pass
layout["Danielle_Smith"] = [288, 1068, 602, 1212]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_15_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-17/27_text_General_Admission.png
try:
    _c27 = get_crop(27, 75, 72)
    canvas.paste(_c27, (249, 2588), _c27)
except Exception:
    pass
layout["General_Admission"] = [249, 2588, 324, 2660]
