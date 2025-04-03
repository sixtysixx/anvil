<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
  <xsl:template match="/">
    <html>
      <body>
        <h2>XSLT Transformation Example</h2>
        <table border="1">
          <tr bgcolor="grey">
            <th>Title</th>
            <th>URL</th>
          </tr>
          <xsl:for-each select="channel">
            <xsl:for-each select="item">
            <tr>
              <td><xsl:value-of select="title"></xsl:value-of></td>
              <td><xsl:value-of select="link"></xsl:value-of></td>
            </tr>
            </xsl:for-each>
          </xsl:for-each>
        </table>
      </body>
    </html>
  </xsl:template>
</xsl:stylesheet>