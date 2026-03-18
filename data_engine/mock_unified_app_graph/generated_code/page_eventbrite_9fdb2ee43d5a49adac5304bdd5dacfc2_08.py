# page_id: page_eventbrite_9fdb2ee43d5a49adac5304bdd5dacfc2_08
# screenshot: 2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-10.png
# step_index: 8/8
# task: Open Eventbrite. Look up 'Pet' events. Filter by events happening this weekend. Select the third non-promoted event from the results - how much are the tickets for the event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas and draw are provided (1440x2960). This script paints the background layout and structural UI elements.
w, h = canvas.size

# Colors
bg_color = (250, 250, 250)         # soft off-white page background
status_bar_color = (200, 200, 200) # muted grey status bar
image_top_colors = [(85, 54, 119), (134, 83, 168), (194, 153, 211)]  # gradient band for header image area
image_overlay = (0, 0, 0, 40)      # translucent overlay for header
pill_yellow = (255, 244, 179)      # soft yellow pill (sold-out background)
card_bg = (255, 255, 255)          # white cards
muted_divider = (230, 230, 234)    # subtle divider lines
tag_bg = (241, 244, 248)           # light tag background
footer_bg = (246, 246, 249)        # footer panel background
shadow_color = (0, 0, 0, 24)       # soft shadow (semi-transparent)

# Helper to draw rounded rect with optional shadow
def rounded_rect(xy, radius, fill, outline=None, outline_width=0, shadow=False):
    x0, y0, x1, y1 = xy
    if shadow:
        # draw a subtle shadow offset
        sx0, sy0, sx1, sy1 = x0+6, y0+8, x1+6, y1+10
        try:
            draw.rounded_rectangle([sx0, sy0, sx1, sy1], radius=radius, fill=shadow_color)
        except Exception:
            # fallback: draw simple rectangle shadow
            draw.rectangle([sx0, sy0, sx1, sy1], fill=(0,0,0,10))
    draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=fill, outline=outline, width=outline_width)

# Fill page background
draw.rectangle([0, 0, w, h], fill=bg_color)

# Status bar (top)
status_h = 84
draw.rectangle([0, 0, w, status_h], fill=status_bar_color)

# Header/image banner area (gradient fill)
img_top = status_h
img_h = 520
img_bottom = img_top + img_h
# Vertical gradient
for i in range(img_h):
    # interpolate between three colors
    t = i / max(1, img_h-1)
    if t < 0.5:
        # blend between first and second
        mix = t / 0.5
        c1 = image_top_colors[0]
        c2 = image_top_colors[1]
    else:
        mix = (t - 0.5) / 0.5
        c1 = image_top_colors[1]
        c2 = image_top_colors[2]
    r = int(c1[0] * (1-mix) + c2[0] * mix)
    g = int(c1[1] * (1-mix) + c2[1] * mix)
    b = int(c1[2] * (1-mix) + c2[2] * mix)
    draw.line([(0, img_top + i), (w, img_top + i)], fill=(r, g, b))

# Subtle dark overlay near top of the image banner (to emulate nav overlay area)
overlay_height = 120
overlay_box = (0, img_top, w, img_top + overlay_height)
overlay_img = Image.new("RGBA", (w, overlay_height), image_overlay)
canvas.paste(overlay_img, (0, img_top), overlay_img)

# Divider under header image
draw.line([(48, img_bottom + 8), (w-48, img_bottom + 8)], fill=muted_divider, width=1)

# Yellow pill banner (Sold out / status) - rounded rectangle background only
pill_x0 = 48
pill_x1 = w - 48
pill_y0 = img_bottom + 28
pill_h = 88
pill_radius = 22
rounded_rect([pill_x0, pill_y0, pill_x1, pill_y0 + pill_h], radius=pill_radius, fill=pill_yellow)

# Main content area starts below the pill
content_top = pill_y0 + pill_h + 36

# Large page title area left blank (we intentionally do NOT draw text)
# Add a light divider under the title area
title_div_y = content_top + 160
draw.line([(48, title_div_y), (w-48, title_div_y)], fill=muted_divider, width=1)

# Artist / Host profile card (rounded white card with subtle shadow)
card_x0 = 48
card_x1 = w - 48
card_y0 = title_div_y + 28
card_h = 180
card_y1 = card_y0 + card_h
rounded_rect([card_x0, card_y0, card_x1, card_y1], radius=28, fill=card_bg, shadow=True)

# Small subtle divider below the profile card
draw.line([(48, card_y1 + 28), (w-48, card_y1 + 28)], fill=muted_divider, width=1)

# Details list area (location, duration, refund) - keep whitespace, draw small separators
list_top = card_y1 + 48
# Draw icon-aligned left separators (just visual subtle bullets placeholders - NO icons)
sep_x = 48 + 64  # left alignment where icons would sit
for i in range(3):
    y = list_top + i*120
    # small circular placeholder background (very muted) to indicate spacing for icons (no icon drawn)
    draw.ellipse([sep_x-22, y-18, sep_x+22, y+18], fill=(245,245,247))

# Light horizontal divider before About section
about_div_y = list_top + 3*120 + 16
draw.line([(48, about_div_y), (w-48, about_div_y)], fill=muted_divider, width=1)

# "About this event" section background and tag pill (only background shapes)
about_top = about_div_y + 28
# Tag pill (category tag background)
tag_x0 = 48
tag_x1 = 740
tag_y0 = about_top
tag_h = 64
rounded_rect([tag_x0, tag_y0, tag_x1, tag_y0 + tag_h], radius=32, fill=tag_bg)

# Body area (light area reserved for description) - do not draw text
desc_top = tag_y0 + tag_h + 28
desc_h = 220
# Keep background same as page; draw very light bounding line to indicate the content region
draw.rectangle([48, desc_top, w-48, desc_top + desc_h], outline=muted_divider, width=1)

# Footer "Sales ended" bar area at bottom
footer_h = 160
footer_y0 = h - footer_h
draw.rectangle([0, footer_y0, w, h], fill=footer_bg)
# top divider of footer
draw.line([(0, footer_y0), (w, footer_y0)], fill=muted_divider, width=1)

# Right-side details button background in footer (outline only, to be overlaid by actual 'Details' button content)
btn_w = 520
btn_h = 112
btn_x1 = w - 48
btn_x0 = btn_x1 - btn_w
btn_y0 = footer_y0 + 24
btn_y1 = btn_y0 + btn_h
draw.rounded_rectangle([btn_x0, btn_y0, btn_x1, btn_y1], radius=12, outline=(160,160,170), width=6, fill=(255,255,255))

# Left side "Sales ended" area (background subtle)
sales_x0 = 48
sales_x1 = btn_x0 - 24
sales_y0 = footer_y0 + 24
sales_y1 = sales_y0 + btn_h
# Draw a slightly darker white panel behind the text area (no text)
rounded_rect([sales_x0, sales_y0, sales_x1, sales_y1], radius=12, fill=(250,249,251))

# Additional subtle separators across page
for y in [title_div_y + 320, about_div_y + 320]:
    draw.line([(48, y), (w-48, y)], fill=(245,245,247), width=1)

# Final fine top accent line under status bar and above image (thin)
draw.line([(0, status_h), (w, status_h)], fill=(220,220,220), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_08_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-10/00_icon_Follow.png
try:
    _c0 = get_crop(0, 331, 144)
    canvas.paste(_c0, (1013, 1331), _c0)
except Exception:
    pass
layout["Follow"] = [1013, 1331, 1344, 1475]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_08_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-10/01_icon_Details.png
try:
    _c1 = get_crop(1, 522, 144)
    canvas.paste(_c1, (822, 2768), _c1)
except Exception:
    pass
layout["Details"] = [822, 2768, 1344, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_08_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-10/02_icon_More.png
try:
    _c2 = get_crop(2, 144, 144)
    canvas.paste(_c2, (1116, 108), _c2)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_08_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-10/03_icon_Share.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1260, 108), _c3)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_08_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-10/04_icon_4.48.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (36, 108), _c4)
except Exception:
    pass
layout["4.48"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_08_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-10/05_icon_2_hrs_30_mins.png
try:
    _c5 = get_crop(5, 310, 74)
    canvas.paste(_c5, (119, 1704), _c5)
except Exception:
    pass
layout["2_hrs_30_mins"] = [119, 1704, 429, 1778]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_08_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-10/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 58, 67)
    canvas.paste(_c6, (1316, 0), _c6)
except Exception:
    pass
layout["icon_6"] = [1316, 0, 1374, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_08_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-10/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 101, 66)
    canvas.paste(_c7, (1213, 0), _c7)
except Exception:
    pass
layout["icon_7"] = [1213, 0, 1314, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_08_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-10/08_icon_Join_Artist_Katie_Detrich_owner_of_Welco.png
try:
    _c8 = get_crop(8, 234, 144)
    canvas.paste(_c8, (48, 2468), _c8)
except Exception:
    pass
layout["Join_Artist_Katie_Detrich"] = [48, 2468, 282, 2612]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_08_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-10/09_icon_6.30_PM.png
try:
    _c9 = get_crop(9, 409, 144)
    canvas.paste(_c9, (288, 1291), _c9)
except Exception:
    pass
layout["6.30_PM"] = [288, 1291, 697, 1435]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_08_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-10/10_icon_4.48.png
try:
    _c10 = get_crop(10, 65, 68)
    canvas.paste(_c10, (179, 1), _c10)
except Exception:
    pass
layout["4.48"] = [179, 1, 244, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_08_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-10/11_icon_4.48.png
try:
    _c11 = get_crop(11, 63, 71)
    canvas.paste(_c11, (114, 0), _c11)
except Exception:
    pass
layout["4.48"] = [114, 0, 177, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_08_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-10/12_icon_Hobbies_Special_Interest_._Drawing_Paint.png
try:
    _c12 = get_crop(12, 234, 144)
    canvas.paste(_c12, (48, 2468), _c12)
except Exception:
    pass
layout["Hobbies_&_Special_Interes"] = [48, 2468, 282, 2612]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_08_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-10/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 71, 70)
    canvas.paste(_c13, (306, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [306, 0, 377, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_08_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-10/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 57, 70)
    canvas.paste(_c14, (246, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [246, 0, 303, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_08_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-10/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 52, 70)
    canvas.paste(_c15, (382, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [382, 0, 434, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_08_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-10/16_icon_Bubba_s_33.png
try:
    _c16 = get_crop(16, 144, 144)
    canvas.paste(_c16, (96, 1330), _c16)
except Exception:
    pass
layout["Bubba's_33"] = [96, 1330, 240, 1474]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_08_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-10/17_text_4.48.png
try:
    _c17 = get_crop(17, 89, 43)
    canvas.paste(_c17, (22, 15), _c17)
except Exception:
    pass
layout["4.48"] = [22, 15, 111, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_08_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-10/18_text_HIstS.png
try:
    _c18 = get_crop(18, 46, 27)
    canvas.paste(_c18, (765, 331), _c18)
except Exception:
    pass
layout["HIstS"] = [765, 331, 811, 358]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_08_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-10/19_text_Sunday_April_28.png
try:
    _c19 = get_crop(19, 416, 73)
    canvas.paste(_c19, (38, 929), _c19)
except Exception:
    pass
layout["Sunday,_April_28"] = [38, 929, 454, 1002]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_08_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-10/20_text_6.30_PM.png
try:
    _c20 = get_crop(20, 210, 55)
    canvas.paste(_c20, (483, 934), _c20)
except Exception:
    pass
layout["6.30_PM"] = [483, 934, 693, 989]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_08_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-10/21_text_Paint_Your_Pet_Glen_Burnie_Bubba_s_33.png
try:
    _c21 = get_crop(21, 409, 144)
    canvas.paste(_c21, (288, 1291), _c21)
except Exception:
    pass
layout["Paint_Your_Pet!_Glen_Burn"] = [288, 1291, 697, 1435]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_08_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-10/22_text_with_Artist_Katie_Detrichl.png
try:
    _c22 = get_crop(22, 409, 144)
    canvas.paste(_c22, (288, 1291), _c22)
except Exception:
    pass
layout["with_Artist_Katie_Detrich"] = [288, 1291, 697, 1435]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_08_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-10/23_text_Artist_Katie_Detrich.png
try:
    _c23 = get_crop(23, 409, 144)
    canvas.paste(_c23, (288, 1291), _c23)
except Exception:
    pass
layout["Artist_Katie_Detrich"] = [288, 1291, 697, 1435]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_08_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-10/24_text_1.7k_Followers.png
try:
    _c24 = get_crop(24, 409, 144)
    canvas.paste(_c24, (288, 1291), _c24)
except Exception:
    pass
layout["1.7k_Followers"] = [288, 1291, 697, 1435]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_08_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-10/25_text_Bubba_s_33.png
try:
    _c25 = get_crop(25, 248, 54)
    canvas.paste(_c25, (139, 1605), _c25)
except Exception:
    pass
layout["Bubba's_33"] = [139, 1605, 387, 1659]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_08_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-10/26_text_Refund_policy.png
try:
    _c26 = get_crop(26, 299, 64)
    canvas.paste(_c26, (138, 1821), _c26)
except Exception:
    pass
layout["Refund_policy"] = [138, 1821, 437, 1885]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_08_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-10/27_text_No_refunds.png
try:
    _c27 = get_crop(27, 210, 45)
    canvas.paste(_c27, (142, 1914), _c27)
except Exception:
    pass
layout["No_refunds"] = [142, 1914, 352, 1959]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_08_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-10/28_text_About_this_event.png
try:
    _c28 = get_crop(28, 453, 64)
    canvas.paste(_c28, (44, 2119), _c28)
except Exception:
    pass
layout["About_this_event"] = [44, 2119, 497, 2183]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_08_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-10/29_text_Sales_ended.png
try:
    _c29 = get_crop(29, 274, 55)
    canvas.paste(_c29, (90, 2814), _c29)
except Exception:
    pass
layout["Sales_ended"] = [90, 2814, 364, 2869]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_08_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-10/30_text_Due.png
try:
    _c30 = get_crop(30, 54, 34)
    canvas.paste(_c30, (764, 286), _c30)
except Exception:
    pass
layout["'Due"] = [764, 286, 818, 320]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_08_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-10/31_clickable_Location.png
try:
    _c31 = get_crop(31, 1344, 144)
    canvas.paste(_c31, (48, 1558), _c31)
except Exception:
    pass
layout["Location"] = [48, 1558, 1392, 1702]
