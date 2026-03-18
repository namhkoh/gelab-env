# page_id: page_eventbrite_97cc452004d64d67bb4c5b51ffb8f6d4_07
# screenshot: 2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-9.png
# step_index: 7/7
# task: Open Eventbrite. Search Business event. Select the first one that is not promoted. Share to google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw structured background for the mobile UI (no icons/text)
# Uses provided variables: canvas (PIL Image), draw (ImageDraw), font_* (unused)

W, H = canvas.size

# Overall background (very light neutral)
draw.rectangle([(0, 0), (W, H)], fill="#fbfbfd")

# Status bar area (top bar ~60px)
status_h = 60
draw.rectangle([(0, 0), (W, status_h)], fill="#9da3a9")

# Subtle bottom line under status bar
draw.line([(0, status_h), (W, status_h)], fill="#e1e3e6", width=1)

# Large hero/banner image area (will have actual image pasted on top).
# We paint a dark multi-band background to match screenshot's dark photo area.
hero_top = status_h
hero_bottom = 520
bands = 10
for i in range(bands):
    # interpolate between two dark colors to create a vertical banded gradient
    t = i / max(1, bands - 1)
    # dark bluish to deep charcoal
    r = int((1 - t) * 36 + t * 20)
    g = int((1 - t) * 40 + t * 18)
    b = int((1 - t) * 48 + t * 24)
    y0 = hero_top + int((hero_bottom - hero_top) * (i / bands))
    y1 = hero_top + int((hero_bottom - hero_top) * ((i + 1) / bands))
    draw.rectangle([(0, y0), (W, y1)], fill=(r, g, b))

# Soft vertical vignette edges on hero (darker at very edges)
edge_width = 140
for i in range(edge_width):
    alpha = int(12 * (1 - i / edge_width))  # small darkening
    if alpha <= 0:
        continue
    shade = (10 + i // 8, 10 + i // 8, 12 + i // 6)
    # Draw thin vertical lines with slightly changing dark shade to simulate vignette
    draw.line([(i, hero_top), (i, hero_bottom)], fill=shade)
    draw.line([(W - 1 - i, hero_top), (W - 1 - i, hero_bottom)], fill=shade)

# Subtle highlight strip at top of hero (beneath status bar) to suggest glossy image
highlight_h = 16
for i in range(highlight_h):
    t = i / max(1, highlight_h - 1)
    col = (int(240 - 40 * t), int(240 - 40 * t), int(245 - 35 * t))
    draw.line([(0, hero_top + i), (W, hero_top + i)], fill=col)

# Drop shadow under hero (soft horizontal fade)
shadow_top = hero_bottom - 6
for i in range(12):
    y = shadow_top + i
    alpha_factor = int(80 * (1 - i / 12))
    if alpha_factor <= 0:
        continue
    grey = 20 + i  # slightly lighter as it fades
    draw.line([(40, y), (W - 40, y)], fill=(grey, grey, grey))

# Main content area (white card region starts below hero)
content_top = hero_bottom + 20
# We keep it primarily white but add a very faint tint to differentiate from overall bg
draw.rectangle([(0, content_top), (W, H - 700)], fill="#ffffff")

# Organizer / follow card (rounded rectangle background behind organizer info + Follow button)
# This is a structural background only; the actual "Follow" button and text/icons will be pasted later.
org_left = 36
org_right = W - 36
org_top = 1280  # aligns to approximate positions where organizer card appears
org_bottom = org_top + 160
draw.rounded_rectangle([(org_left, org_top), (org_right, org_bottom)],
                       radius=28, fill="#f6f6f9", outline="#e7e7ee", width=2)

# Slight inner top highlight for organizer card
draw.line([(org_left + 2, org_top + 3), (org_right - 2, org_top + 3)], fill="#ffffff", width=1)

# Horizontal separators between content sections
sep_color = "#e9e9ef"
# separator below organizer/refund area
sep_y1 = org_bottom + 120
draw.line([(36, sep_y1), (W - 36, sep_y1)], fill=sep_color, width=1)

# Another separator further down (section divider)
sep_y2 = sep_y1 + 200
draw.line([(36, sep_y2), (W - 36, sep_y2)], fill=sep_color, width=1)

# Light background card behind "About this event" block area to give structure
about_top = 2040
about_left = 36
about_right = W - 36
about_bottom = about_top + 200
draw.rectangle([(about_left, about_top), (about_right, about_bottom)], fill="#ffffff")
# faint shadow line on top to separate
draw.line([(about_left, about_top), (about_right, about_top)], fill="#f1f1f4", width=3)

# Prevent drawing within bottom reserve area: leave bottom reserved region untouched.
reserved_top = 2324  # do not draw inside y >= reserved_top (reserve button area)
# Provide a faint top divider right above reserved area
divider_y = reserved_top - 24
draw.line([(24, divider_y), (W - 24, divider_y)], fill="#efeef1", width=2)

# Subtle left gutter shadow down the main content column for depth
for i in range(20):
    x = 36 + i
    col = 250 - i  # slightly darker near edge
    draw.line([(x, content_top), (x, reserved_top - 40)], fill=(col, col, col))

# Right gutter subtle shadow
for i in range(12):
    x = W - 36 - i
    col = 250 - i
    draw.line([(x, content_top), (x, reserved_top - 40)], fill=(col, col, col))

# Finished structural background. Actual icons/text/buttons will be pasted on top by the pipeline.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_07_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-9/00_icon_Follow.png
try:
    _c0 = get_crop(0, 331, 144)
    canvas.paste(_c0, (1013, 1290), _c0)
except Exception:
    pass
layout["Follow"] = [1013, 1290, 1344, 1434]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_07_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-9/01_icon_FAIR.png
try:
    _c1 = get_crop(1, 144, 144)
    canvas.paste(_c1, (1116, 108), _c1)
except Exception:
    pass
layout["FAIR"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_07_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-9/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 114, 106)
    canvas.paste(_c2, (987, 2439), _c2)
except Exception:
    pass
layout["icon_2"] = [987, 2439, 1101, 2545]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_07_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-9/03_icon_Share.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1260, 108), _c3)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_07_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-9/04_icon_Few_tickets_left.png
try:
    _c4 = get_crop(4, 431, 86)
    canvas.paste(_c4, (41, 753), _c4)
except Exception:
    pass
layout["Few_tickets_left"] = [41, 753, 472, 839]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_07_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-9/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 106, 103)
    canvas.paste(_c5, (1216, 2442), _c5)
except Exception:
    pass
layout["icon_5"] = [1216, 2442, 1322, 2545]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_07_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-9/06_icon_Reserve_a_spot.png
try:
    _c6 = get_crop(6, 1440, 636)
    canvas.paste(_c6, (0, 2324), _c6)
except Exception:
    pass
layout["Reserve_a_spot"] = [0, 2324, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_07_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-9/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 93, 103)
    canvas.paste(_c7, (1108, 2442), _c7)
except Exception:
    pass
layout["icon_7"] = [1108, 2442, 1201, 2545]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_07_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-9/08_icon_Business_Professional.png
try:
    _c8 = get_crop(8, 690, 98)
    canvas.paste(_c8, (37, 2167), _c8)
except Exception:
    pass
layout["Business_&_Professional"] = [37, 2167, 727, 2265]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_07_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-9/09_icon_9.40.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (36, 108), _c9)
except Exception:
    pass
layout["9.40"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_07_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-9/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 52, 62)
    canvas.paste(_c10, (1319, 2), _c10)
except Exception:
    pass
layout["icon_10"] = [1319, 2, 1371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_07_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-9/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 89, 60)
    canvas.paste(_c11, (1216, 2), _c11)
except Exception:
    pass
layout["icon_11"] = [1216, 2, 1305, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_07_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-9/12_icon_New_York_Tech_Career_Fair_Exclusive.png
try:
    _c12 = get_crop(12, 354, 144)
    canvas.paste(_c12, (144, 1250), _c12)
except Exception:
    pass
layout["New_York_Tech_Career_Fair"] = [144, 1250, 498, 1394]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_07_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-9/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 52, 58)
    canvas.paste(_c13, (316, 6), _c13)
except Exception:
    pass
layout["icon_13"] = [316, 6, 368, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_07_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-9/14_icon_New_York_Tech_Career_Fair_Exclusive.png
try:
    _c14 = get_crop(14, 354, 144)
    canvas.paste(_c14, (144, 1250), _c14)
except Exception:
    pass
layout["New_York_Tech_Career_Fair"] = [144, 1250, 498, 1394]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_07_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-9/15_icon_9.40.png
try:
    _c15 = get_crop(15, 52, 61)
    canvas.paste(_c15, (184, 2), _c15)
except Exception:
    pass
layout["9.40"] = [184, 2, 236, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_07_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-9/16_icon_Tech.png
try:
    _c16 = get_crop(16, 144, 144)
    canvas.paste(_c16, (1116, 108), _c16)
except Exception:
    pass
layout["Tech"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_07_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-9/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 42, 53)
    canvas.paste(_c17, (1272, 7), _c17)
except Exception:
    pass
layout["icon_17"] = [1272, 7, 1314, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_07_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-9/18_icon_9.40.png
try:
    _c18 = get_crop(18, 52, 64)
    canvas.paste(_c18, (117, 1), _c18)
except Exception:
    pass
layout["9.40"] = [117, 1, 169, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_07_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-9/19_icon_The_organizer_will_review_refund_request.png
try:
    _c19 = get_crop(19, 1344, 144)
    canvas.paste(_c19, (48, 1517), _c19)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1517, 1392, 1661]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_07_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-9/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 53, 58)
    canvas.paste(_c20, (248, 4), _c20)
except Exception:
    pass
layout["icon_20"] = [248, 4, 301, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_07_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-9/21_text_9.40.png
try:
    _c21 = get_crop(21, 91, 43)
    canvas.paste(_c21, (20, 15), _c21)
except Exception:
    pass
layout["9.40"] = [20, 15, 111, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_07_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-9/22_text_About_this_event.png
try:
    _c22 = get_crop(22, 454, 61)
    canvas.paste(_c22, (45, 2080), _c22)
except Exception:
    pass
layout["About_this_event"] = [45, 2080, 499, 2141]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_07_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-9/23_text_RSVP.png
try:
    _c23 = get_crop(23, 130, 52)
    canvas.paste(_c23, (116, 2451), _c23)
except Exception:
    pass
layout["RSVP"] = [116, 2451, 246, 2503]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_07_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-9/24_text_Free.png
try:
    _c24 = get_crop(24, 105, 48)
    canvas.paste(_c24, (116, 2599), _c24)
except Exception:
    pass
layout["Free"] = [116, 2599, 221, 2647]
