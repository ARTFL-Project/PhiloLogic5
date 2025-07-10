<template>
    <div class="modal-dialog modal-lg modal-dialog-scrollable" role="dialog" aria-labelledby="search-tips-title"
        aria-describedby="search-tips-description">
        <div class="modal-content p-2">
            <div class="modal-header">
                <h2 class="modal-title h5" id="search-tips-title">{{ $t('searchTips.title') }}</h2>
                <button type="button" class="btn-close" data-bs-dismiss="modal" :aria-label="$t('common.closeModal')"
                    @click="$event.target.blur()">
                </button>
            </div>
            <div class="modal-body">
                <p id="search-tips-description">
                    {{ $t('searchTips.description') }}:
                </p>

                <section aria-labelledby="basic-operators-heading">
                    <h3 id="basic-operators-heading" class="h5">{{ $t('searchTips.basicOperators') }}</h3>
                    <ul role="list">
                        <li role="listitem">
                            {{ $t('searchTips.plainToken') }}
                            <code class="code-block" aria-label="code example">token</code>
                        </li>
                        <li role="listitem">
                            {{ $t('searchTips.quotedToken') }}
                            <code class="code-block" aria-label="code example">"token"</code>
                        </li>
                        <li role="listitem">
                            {{ $t('searchTips.range') }}
                            <code class="code-block" aria-label="code example">a-f</code>
                        </li>
                        <li role="listitem">
                            {{ $t('searchTips.booleanOr') }}
                            <code class="code-block" aria-label="code example">|</code>, {{ $t('common.example') }}
                            <code class="code-block" aria-label="code example">token | word</code>
                        </li>
                        <li role="listitem">
                            {{ $t('searchTips.booleanNot') }}
                            <code class="code-block" aria-label="code example">NOT</code>, {{ $t('common.example') }}
                            <code class="code-block" aria-label="code example">token.* NOT tokens</code>
                        </li>
                    </ul>
                </section>

                <p>{{ $t('searchTips.syntaxNote') }}</p>

                <section aria-labelledby="word-searches-heading">
                    <h3 id="word-searches-heading" class="h5" style="margin-top: 20px">{{ $t('searchTips.wordSearches')
                    }}</h3>
                    <p class="mb-0">{{ $t('searchTips.wordSearchDescription') }}</p>
                    <ol role="list">
                        <li role="listitem">{{ $t('searchTips.plainTerms') }}</li>
                        <li role="listitem">{{ $t('searchTips.quotedTerms') }}</li>
                        <li role="listitem">{{ $t('searchTips.rangeNote') }}</li>
                        <li role="listitem">
                            <code class="code-block" aria-label="code example">OR</code> {{
                                $t('searchTips.orDescription') }}
                        </li>
                        <li role="listitem">
                            <code class="code-block" aria-label="code example">NOT</code> {{
                                $t('searchTips.notDescription') }}
                            <code class="code-block" aria-label="code example">a.* NOT abalone</code> {{
                                $t('searchTips.isLegal') }},
                            <code class="code-block" aria-label="code example">NOT a.*</code> {{
                                $t('searchTips.isIllegal') }}
                        </li>
                    </ol>
                </section>

                <section aria-labelledby="morphology-heading">
                    <h4 id="morphology-heading" class="h5">{{ $t('searchTips.morphologyTitle') }}</h4>
                    <ol role="list">
                        <li role="listitem">
                            {{ $t('searchTips.lemmaSearch') }}
                            <code class="code-block" aria-label="code example">lemma:</code>
                            {{ $t('searchTips.lemmaExample') }}
                            <code class="code-block" aria-label="code example">lemma:have</code>.
                            {{ $t('searchTips.regexPermitted') }}
                            <code class="code-block" aria-label="code example">lemma:constitut.*</code>.
                        </li>
                        <li role="listitem">
                            {{ $t('searchTips.wordAttribute') }}
                            <code class="code-block" aria-label="code example">word:attribute:attribute</code>
                            {{ $t('searchTips.wordAttributeExample') }}
                            <code class="code-block" aria-label="code example">love:pos:NOUN</code>.
                        </li>
                        <li role="listitem">
                            {{ $t('searchTips.combineLemma') }}
                            <code class="code-block" aria-label="code example">lemma:love:pos:NOUN</code>.
                        </li>
                        <li role="listitem">
                            {{ $t('searchTips.multipleAttributes') }}
                            <code class="code-block" aria-label="code example">charles:pos:PROPN:ner:PERS</code>.
                        </li>
                    </ol>
                </section>

                <section aria-labelledby="metadata-searches-heading">
                    <h3 id="metadata-searches-heading" class="h5" style="margin-top: 20px">{{
                        $t('searchTips.metadataSearches') }}
                    </h3>
                    <p class="mb-0">{{ $t('searchTips.metadataDescription') }}</p>
                    <ol role="list">
                        <li role="listitem">{{ $t('searchTips.metadataPlainTokens') }}</li>
                        <li role="listitem">{{ $t('searchTips.metadataQuotedTokens') }}</li>
                        <li role="listitem">{{ $t('searchTips.metadataRange') }}</li>
                        <li role="listitem">
                            <code class="code-block" aria-label="code example">OR</code> {{ $t('searchTips.metadataOr')
                            }}
                        </li>
                        <li role="listitem">
                            <code class="code-block" aria-label="code example">NOT</code> {{
                                $t('searchTips.metadataNot') }}
                            <code class="code-block" aria-label="code example">contrat NOT social</code> {{
                                $t('searchTips.isLegal') }},
                            <code class="code-block" aria-label="code example">NOT rousseau</code>
                        </li>
                    </ol>
                </section>

                <section aria-labelledby="regexp-heading">
                    <h3 id="regexp-heading" class="h5" style="margin-top: 20px">{{ $t('searchTips.regexpSyntax') }}</h3>
                    <p class="mb-0">
                        {{ $t('searchTips.regexpDescription') }}
                        <a href="http://www.gnu.org/software/findutils/manual/html_node/find_html/egrep-regular-expression-syntax.html#egrep-regular-expression-syntax"
                            target="_blank" rel="noopener noreferrer" :aria-label="$t('searchTips.egrepLinkLabel')">
                            {{ $t('searchTips.egrepLink') }}
                            <span class="visually-hidden">{{ $t('common.opensInNewWindow') }}</span>
                        </a>:
                    </p>
                    <ol role="list">
                        <li role="listitem">
                            {{ $t('searchTips.dotCharacter') }}
                            <code class="code-block" aria-label="code example">.</code> {{
                                $t('searchTips.dotDescription') }}
                        </li>
                        <li role="listitem">
                            {{ $t('searchTips.bracketExpressions') }}
                            <code class="code-block" aria-label="code example">[aeiou]</code> {{ $t('common.or') }}
                            <code class="code-block" aria-label="code example">[a-z]</code>, {{
                                $t('searchTips.bracketNote') }}
                        </li>
                        <li role="listitem">
                            <code class="code-block" aria-label="code example">*</code> {{
                                $t('searchTips.asteriskDescription') }}
                        </li>
                        <li role="listitem">
                            <code class="code-block" aria-label="code example">+</code> {{
                                $t('searchTips.plusDescription') }}
                        </li>
                        <li role="listitem">
                            <code class="code-block" aria-label="code example">?</code> {{
                                $t('searchTips.questionDescription') }}
                        </li>
                    </ol>
                    <p class="mb-0">
                        {{ $t('searchTips.wildcardNote') }}
                        <code class="code-block" aria-label="code example">.*</code> {{
                            $t('searchTips.wildcardDescription') }}
                        <code class="code-block" aria-label="code example">*</code> {{
                            $t('searchTips.traditionalWildcard') }}
                    </p>
                </section>
            </div>
        </div>
    </div>
</template>
<script>
export default {
    name: "SearchTips",
};
</script>
<style lang="scss" scoped>
@import "../assets/styles/theme.module.scss";

li {
    list-style-type: disc;
}

.code-block {
    background-color: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 0.25rem;
    padding: 0.125rem 0.25rem;
    font-family: 'Courier New', Courier, monospace;
    font-size: 0.875em;
}

/* Focus styles for links */
a:focus {
    outline: 2px solid $link-color;
    outline-offset: 2px;
}

section {
    margin-bottom: 1.5rem;
}

.h5 {
    font-weight: 600;
}
</style>