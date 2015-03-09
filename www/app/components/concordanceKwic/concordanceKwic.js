philoApp.controller('concordanceKwicCtrl', ['$scope', '$rootScope', '$location', 'request', 'URL', function($scope, $rootScope, $location, request, URL) {
                                                
    $rootScope.formData = angular.copy($location.search());
    if ($rootScope.formData.q === "" && $rootScope.report !== "bibliography") {
        var urlString = URL.objectToUrlString($rootScope.formData, {report: "bibliography"});
        $location.url(urlString);
    }
    
    request.report($rootScope.formData).then(function(results) {
        $scope.results = results.data;
        $scope.description = $scope.results.description;
    })

    $scope.showFullBiblio = function(event) {
        target = $(event.currentTarget).find('.full_biblio');
        target.addClass('show');
    }
    $scope.hideFullBiblio = function(event) {
        target = $(event.currentTarget).find('.full_biblio');
        target.removeClass('show');
    }
    $rootScope.frequencyResults = []; // TODO: move this out of rootScope
    $scope.resultsContainerWidth = "";
    $scope.sidebarWidth = '';
    $scope.$watch('frequencyResults', function(frequencyResults) {
        if (frequencyResults.length > 0) {
            $scope.resultsContainerWidth = "col-xs-8";
            $scope.sidebarWidth = "col-xs-4";
        } else {
            $scope.resultsContainerWidth = "";
            $scope.sidebarWidth = "";
        }
    });
    
    $scope.goToPage = function(start, end) {
        $rootScope.formData.start = start;
        $rootScope.formData.end = end;
        $("body").velocity('scroll', {duration: 200, easing: 'easeOutCirc', offset: 0, complete: function() {
            $rootScope.results = {};
        }});
        $location.url(URL.objectToUrlString($rootScope.formData));
    }
    
    $scope.switchTo = function(report) {
        $('#report label').removeClass('active');
        $('#' + report).addClass('active');
        $location.url(URL.objectToUrlString($rootScope.formData, {report: report}));
    }
    
    $scope.selectedFacet = '';
    $scope.selectFacet = function(facetObj) {
        $scope.selectedFacet = facetObj;
    }
    
    $scope.removeSidebar = function() {
        $scope.frequencyResults = [];
        $('#selected-sidebar-option').data('interrupt', true);
        $scope.selectedFacet = '';
    }
}]);