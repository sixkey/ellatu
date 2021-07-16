import re



matches = re.split(r'!(\[[^\]]*\]\[[^\]]*\])', 'Hello ![image][local.png] that was an image')
images = []
final_message = ""
for index, part in enumerate(matches):
    if index % 2 == 0:
        final_message += part
    else:
        worlds = part[1:-1].split('][', maxsplit=1)
        images.append(worlds)
        final_message += f"_[{len(images)}]_"

print(final_message)
print(images)
