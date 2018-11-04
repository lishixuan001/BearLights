from django.db import models
from django.contrib.auth.models import User
import os

"""
File
"""
class Voice(models.Model):
    # Basic document data.
    # file = models.FileField(verbose_name="file", upload_to='voice/%Y/%m/%d/',
    #                         help_text="This is the voice audio file content")
    file = models.BinaryField(default=None)
    type = models.CharField(default="application/octet-stream")
    # Meta-information.
    added_by = models.ForeignKey(User, verbose_name="added by", null=True, blank=True, editable=False, on_delete=True)
    added_at = models.DateTimeField("added at", auto_now_add=True)
    updated_at = models.DateTimeField("recently changed at", auto_now=True)

    def __unicode__(self):
        return os.path.basename(self.file.name)

    def get_filename(self):
        return os.path.basename(self.file.name)